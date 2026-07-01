SELECT TOP 1
    id,
    appointment_date,
    doctor_name,
    specialty_name
FROM appointments
WHERE (:doctor_name IS NULL OR LOWER(COALESCE(doctor_name, '')) LIKE :doctor_name)
  AND (:specialty_name IS NULL OR LOWER(COALESCE(specialty_name, '')) LIKE :specialty_name)
  AND (:appointment_date IS NULL OR appointment_date = :appointment_date)
ORDER BY id;
