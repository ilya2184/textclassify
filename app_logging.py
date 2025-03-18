import datetime

def writelog(message):
    current_time = datetime.datetime.now(datetime.timezone.utc)
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S %z]")
    print(formatted_time, message)