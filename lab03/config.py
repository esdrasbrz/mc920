import numpy as np

ROW_CLOSING_SELEM=np.ones((1, 100))
COLUMN_CLOSING_SELEM=np.ones((200, 1))
POS_CLOSING_SELEM=np.ones((1,30))
WORDS_DILATION_SELEM=np.ones((6, 1))
WORDS_CLOSING_SELEM=np.ones((1, 10))

TEXT_AREA_RATIO_THR = (.5, .9)
TEXT_TRANS_RATIO_THR = (0, .1)
