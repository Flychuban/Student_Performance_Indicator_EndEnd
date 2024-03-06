import sys

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    error_message = f"Error ocured in python script name {exc_tb.tb_frame.f_code.co_filename} \n line number {exc_tb.tb_lineno} \n error message {str(error)} \n"
    return error_message    

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(self.error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message