import logging

class MemoryFilter(logging.Filter):
    def filter(self, record):
        return 'cuMemFree' not in record.getMessage()



def setup_logging(log_file, level, include_host=False):
    logging.getLogger('numba').setLevel(logging.WARNING)
    # 新增這行
    logging.getLogger('nvfuser').setLevel(logging.ERROR)

    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(MemoryFilter())
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(MemoryFilter())
        logging.root.addHandler(file_handler)
        
    
    
    #logging.getLogger('numba').setLevel(logging.WARNING)