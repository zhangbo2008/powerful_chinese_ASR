#!/usr/bin/python
# -*- coding: UTF-8 -*-

def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b  # 使用 yield
        # print b
        a, b = b, a + b
        n = n + 1


a=fab(5)  # 理解带yield的函数返回的是一个迭代器. 我们用next进行遍历. 迭代器就是当你调用next的时候他才开始计算第一个元素,再调用时候算第二个元素. 然后每一次调用next就会从上一次停的地方继续跑. 一直跑到yield再停下.
a.__next__()
a.__next__()
a.__next__()