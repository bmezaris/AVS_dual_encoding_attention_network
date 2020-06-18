import os
import logging

ROOT_PATH=os.path.join(os.environ['HOME'], 'Desktop', 'CERTH_VisualSearch_dualDense')
# ROOT_PATH=os.path.join('/media', 'dgalanop', 'Lexar_ssd', 'CERTH_VisualSearch_dualDense')

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)

