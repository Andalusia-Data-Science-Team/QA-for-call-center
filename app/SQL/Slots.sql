SET STATISTICS IO ON;
SET STATISTICS TIME ON;
SET NOCOUNT ON;
-- Use READ UNCOMMITTED to prevent query from waiting on locks
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- === 1. DECLARE PARAMETERS ===
DECLARE 
    @ReportDate DATE,
	@Specialty NVARCHAR(100),
	@Doctor NVARCHAR(100),
    @PatientPhone NVARCHAR(100),
	@PatientName NVARCHAR(800);



-- === 2. ASSIGN PARAMETERS ===

-- API Assignment (Values injected by the API at runtime)
SET @ReportDate = :ReportDate;
SET @Specialty = :Specialty;
SET @Doctor = :Doctor;
SET @PatientPhone = :PatientPhone;
SET @PatientName = :PatientName;

-- === 3. DATE RANGES (SARGable) ===
DECLARE @StartOfDay DATETIME = @ReportDate;
DECLARE @EndOfDay DATETIME = DATEADD(DAY, 1, @ReportDate);

-- === 4. CLEANUP TEMP TABLES ===
DROP TABLE IF EXISTS #ApptAgg, #SchedIntervals, #WorkIntervals, #WorkMinutesAgg, #SlotsSummary, #PatientHistory, #ActualWorkingHours;

-- === 5. CREATE TEMP TABLES ===

-- Step 1: Combine ApptAgg and ApptTimes for efficiency
-- This is our primary data pull for the day.
SELECT
    ap.PhysicianID,
    MAX(p.PhysicianEnName) AS Doctor,
	MAX(p.PhysicianArName) AS ArDoctor,
    MAX(ap.SpecialtyEnName) AS Specialty,
	MAX(ap.SpecialtyArName) AS ArSpecialty,
    @ReportDate AS ApptDate,
    COUNT(DISTINCT ap.PatientID) AS TotalPatients,
    MIN(ap.StartDateTime) AS FirstAppt,
    MAX(ap.EndDateTime) AS LastAppt
INTO #ApptAgg
FROM OPD.BK_Appointment ap
INNER JOIN [OPD].[BK_PatternInstance] pl ON ap.PatternInstanceID = pl.ID
INNER JOIN [OPD].PHS_OPDPattern p ON p.ID = pl.PatternID AND p.PhysicianID = pl.PhysicianID
WHERE ap.StartDateTime >= @StartOfDay AND ap.StartDateTime < @EndOfDay 
  AND ap.StatusID NOT IN (6,7)
  AND p.IsDeleted = 0
  AND (p.EndDate IS NULL OR CONVERT(date, pl.StartDateTime) BETWEEN p.StartDate AND p.EndDate)
  --AND (@Specialty IS NULL OR ap.SpecialtyArName = @Specialty)
  AND (@Doctor IS NULL OR p.PhysicianArName LIKE @Doctor)
GROUP BY ap.PhysicianID;
CREATE CLUSTERED INDEX IX_ApptAgg ON #ApptAgg (PhysicianID);

-- Step 2: Scheduled pattern intervals
-- We optimize the slow EXISTS clause by joining to #ApptAgg. If a doctor is in #ApptAgg, they have an appointment.
SELECT
    ps.PhysicianID,
    ps.StartDateTime,
    ps.EndDateTime,
    ps.ID AS PatternInstanceID,
	p.AllowOverBooking,
	ps.PatternID
INTO #SchedIntervals
FROM OPD.BK_PatternInstance ps
INNER JOIN [OPD].PHS_OPDPattern p ON p.ID = ps.PatternID AND p.PhysicianID = ps.PhysicianID
-- This join replaces the slow EXISTS clause
INNER JOIN #ApptAgg a ON ps.PhysicianID = a.PhysicianID 
WHERE ps.StartDateTime >= @StartOfDay AND ps.StartDateTime < @EndOfDay
  AND p.IsDeleted = 0
  AND (p.EndDate IS NULL OR CONVERT(date, ps.StartDateTime) BETWEEN p.StartDate AND p.EndDate);
CREATE CLUSTERED INDEX IX_SchedIntervals ON #SchedIntervals (PhysicianID, PatternInstanceID);

-- Step 3: Work intervals (use schedules when exist, otherwise fallback)
SELECT
    a.PhysicianID,
    a.Doctor,
    a.Specialty,
    a.ApptDate AS WorkDate,
    si.StartDateTime,
    si.EndDateTime
INTO #WorkIntervals
FROM #ApptAgg a
INNER JOIN #SchedIntervals si ON si.PhysicianID = a.PhysicianID
UNION ALL
SELECT
    a.PhysicianID,
    a.Doctor,
    a.Specialty,
    a.ApptDate,
    a.FirstAppt AS StartDateTime,
    a.LastAppt AS EndDateTime
FROM #ApptAgg a
LEFT JOIN #SchedIntervals si ON si.PhysicianID = a.PhysicianID
WHERE si.PhysicianID IS NULL; -- Fallback for doctors with no schedule found
CREATE CLUSTERED INDEX IX_WorkIntervals ON #WorkIntervals (PhysicianID);

-- Step 4: Aggregate minutes per doctor/day
SELECT
    wi.PhysicianID,
    MAX(wi.Doctor) AS Doctor,
    MAX(wi.Specialty) AS Specialty,
    @ReportDate AS WorkDate,
    DATENAME(WEEKDAY, MAX(wi.WorkDate)) AS WeekDayName,
    SUM(DATEDIFF(MINUTE, wi.StartDateTime, wi.EndDateTime)) AS TotalWorkMinutes
INTO #WorkMinutesAgg
FROM #WorkIntervals wi
GROUP BY wi.PhysicianID;
CREATE CLUSTERED INDEX IX_WorkMinutesAgg ON #WorkMinutesAgg (PhysicianID);

-- Step 5: Slots and overbooking summary
SELECT 
    pi.PhysicianID AS DRID,
    SUM(sl.SlotUnit) AS plannedslot,
    SUM(CASE WHEN sl.IsOverbooked = 1 THEN sl.SlotUnit ELSE 0 END) AS OverbookedSlots,
    SUM(CASE WHEN ap.SlotID IS NOT NULL THEN sl.SlotUnit ELSE 0 END) AS ActualBookedSlots
INTO #SlotsSummary
FROM opd.BK_Slot sl
INNER JOIN #SchedIntervals pi ON sl.PatternInstanceID = pi.PatternInstanceID
LEFT JOIN [OPD].[BK_Appointment] ap ON ap.SlotID = sl.ID AND ap.StatusID NOT IN (6,7)
WHERE sl.StartDate = @ReportDate
GROUP BY pi.PhysicianID;
CREATE CLUSTERED INDEX IX_SlotsSummary ON #SlotsSummary (DRID);

-- Step 6: Patient History
SELECT
    pi.PhysicianID,
    p.PhysicianEnName AS Doctor,
    ap.SpecialtyEnName AS Specialty,
	ap.SpecialtyArName AS ArSpecialty,

    ap.PatientID,
	ISNULL(ap.PatientEnName,ap.PatientTempName) AS PatientEnName,
	ISNULL(ap.PatientArName,ap.PatientTempName) AS PatientArName,
    ap.PatientCode,
	ap.PatientMobile,

    sl.StartDate AS SlotDate,
    sl.StartTime AS SlotTime,
    ap.ID AS AppointmentID,
    ap.StatusID
	

INTO #PatientHistory

FROM OPD.BK_Appointment ap

INNER JOIN #SchedIntervals pi
    ON ap.PatternInstanceID = pi.PatternInstanceID

INNER JOIN OPD.BK_Slot sl
    ON sl.ID = ap.SlotID

INNER JOIN OPD.PHS_OPDPattern p
    ON p.ID = pi.PatternID
   AND p.PhysicianID = pi.PhysicianID

WHERE
        ap.StartDateTime < @EndOfDay
    AND ap.StatusID NOT IN (6,7)
	AND (@PatientPhone IS NULL OR ap.PatientMobile Like @PatientPhone);
CREATE CLUSTERED INDEX IX_PatientHistory
ON #PatientHistory(PhysicianID,SlotDate);

-- Step 7: Actual working hours
SELECT 
    va.PhysicianID,
	MAX(vv.MainSpecialityEnName) AS Specialty,
	MAX(vv.MainSpecialityArName) AS ArSpecialty

INTO #ActualWorkingHours
FROM [VisitMgt].[VisitService] vs 
INNER JOIN [VisitMgt].[Visit] vv ON vs.VisitID = vv.ID
INNER JOIN VisitMgt.VisitAppointment va ON va.VisitID = vs.VisitID
WHERE vv.VisitClassificationID = 1 
	AND (
        vs.ClaimDate >= @StartOfDay
    AND vs.ClaimDate < @EndOfDay
    OR
       (vs.ClaimDate IS NULL
        AND vs.CreatedDate >= @StartOfDay
        AND vs.CreatedDate < @EndOfDay)
  )
  AND vs.IsDeleted = 0 
  AND vv.VisitStatusID != 3
GROUP BY va.PhysicianID
OPTION (RECOMPILE);
CREATE CLUSTERED INDEX IX_ActualWorkingHours ON #ActualWorkingHours (PhysicianID);


-- === 8. FINAL SELECT ===
SELECT
    ISNULL(ca.Specialty,awh.Specialty) AS Specialty,
	ISNULL(ph.ArSpecialty,awh.ArSpecialty) AS ArSpecialty,
    ca.Doctor,
	a.ArDoctor,
    ca.WorkDate AS WorkDate,
    ca.WeekDayName AS [Day],
    CASE 
        WHEN ISNULL(a.TotalPatients, 0) > 0 AND (sl.plannedslot - sl.OverbookedSlots) > 0
        THEN CAST((ca.TotalWorkMinutes * 1.0 / (sl.plannedslot - sl.OverbookedSlots)) AS DECIMAL(10,2))
        ELSE NULL 
    END AS [Avg Slot Duration (Min.)],
    sl.plannedslot, 
    sl.OverbookedSlots, 
    (sl.plannedslot - sl.OverbookedSlots) AS PlannedSlots_without_overbooking,
    sl.ActualBookedSlots,
    ca.PhysicianID,
	ph.PatientID,
	ph.PatientEnName,
	ph.PatientArName,
	ph.PatientCode,
	ph.PatientMobile,
	ph.SlotDate,
	ph.SlotTime,
	ph.AppointmentID
FROM #WorkMinutesAgg ca
LEFT JOIN #ApptAgg a ON a.PhysicianID = ca.PhysicianID
LEFT JOIN #SlotsSummary sl ON sl.DRID = ca.PhysicianID
LEFT JOIN #PatientHistory ph ON ph.PhysicianID = ca.PhysicianID
LEFT JOIN #ActualWorkingHours awh ON awh.PhysicianID = ca.PhysicianID
ORDER BY ca.Specialty, ca.Doctor, ca.WorkDate ;

-- === 7. CLEANUP ===
DROP TABLE IF EXISTS #ApptAgg, #SchedIntervals, #WorkIntervals, #WorkMinutesAgg, #SlotsSummary, #PatientHistory, #ActualWorkingHours;
