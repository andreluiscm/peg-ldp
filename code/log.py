from datetime import datetime
import numpy as np


def log_msg(message:np.str_, break_line:np.bool_=False) -> None:
    if break_line:
        print(f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {message}')
    else:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {message}')
