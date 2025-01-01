import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def reset_logs():
    with open("./artifacts/logs/function_trace.log", "w") as log_file:
        log_file.write("")


def trace_function_calls(func):
    def wrapper(*args, **kwargs):
        with open("./artifacts/logs/function_trace.log", "a") as log_file:
            log_file.write(
                f"Function '{func.__name__}' called with arguments: {args} and keyword arguments: {kwargs}\n"
            )
        result = func(*args, **kwargs)
        with open("./artifacts/logs/function_trace.log", "a") as log_file:
            log_file.write(f"Function '{func.__name__}' returned: {result}\n")
            logger.info(f"Function '{func.__name__}' returned: {result}\n")
        return result

    return wrapper
