from eligibility import Beneficiary_api

if __name__ == "__main__":

    """ service = EligibilityService()

    result = service.process_visit(882664)

    print(result["parsed_response"]) """
    result = Beneficiary_api("1110246699")
    print(result)