import os
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(fmt='%(asctime)s %(module)s[%(funcName)s] %(levelname)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

log = logging.getLogger(name='Detect')
log.setLevel(logging.DEBUG)
print_ori = print
print = log.info


def logsys():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(FORMATTER)
    log.addHandler(handler)
    return None


def logfile(log_pth, new_log=False):
    if not os.path.exists(os.path.dirname(log_pth)):
        os.makedirs(os.path.dirname(log_pth))
    if new_log and os.path.isfile(log_pth):
        os.remove(log_pth)
    handler = TimedRotatingFileHandler(log_pth, when='D', encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(FORMATTER)
    log.addHandler(handler)
    return handler

def rmv_handler(handler):
    log.removeHandler(handler)
    return True


# for handler in log.handlers:
#     log.removeHandler(handler)
logsys()

STYLE = {
    'fore':
        {  # 前景色
            'black': 30,  # 黑色
            'red': 31,  # 红色
            'green': 32,  # 绿色
            'yellow': 33,  # 黄色
            'blue': 34,  # 蓝色
            'purple': 35,  # 紫红色
            'cyan': 36,  # 青蓝色
            'white': 37,  # 白色
        },
    'back':
        {  # 背景
            'black': 40,  # 黑色
            'red': 41,  # 红色
            'green': 42,  # 绿色
            'yellow': 43,  # 黄色
            'blue': 44,  # 蓝色
            'purple': 45,  # 紫红色
            'cyan': 46,  # 青蓝色
            'white': 47,  # 白色
        },
    'mode':
        {  # 显示模式
            'mormal': 0,  # 终端默认设置
            'bold': 1,  # 高亮显示
            'underline': 4,  # 使用下划线
            'blink': 5,  # 闪烁
            'invert': 7,  # 反白显示
            'hide': 8,  # 不可见
        },
    'default':
        {
            'end': 0,
        },
}


def UseStyle(string, mode='', fore='', back=''):
    mode = '%s' % STYLE['mode'][mode] if mode in STYLE['mode'] else ''

    fore = '%s' % STYLE['fore'][fore] if fore in STYLE['fore'] else ''

    back = '%s' % STYLE['back'][back] if back in STYLE['back'] else ''

    style = ';'.join([s for s in [mode, fore, back] if s])

    style = '\033[%sm' % style if style else ''

    end = '\033[%sm' % STYLE['default']['end'] if style else ''

    return '%s%s%s' % (style, string, end)


if __name__ == '__main__':
    # logfile('D://DeskTop//xx.txt')
    # print('a')
    print(UseStyle('黑色', fore='red'))
