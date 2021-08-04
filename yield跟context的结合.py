class MyContent:
    def query(self):
        print('query data')
# 需要引入contextmanager
from contextlib import contextmanager

@contextmanager
def make_resource():
    print('first connect to resource')
    yield MyContent()
    print('close resource connection')

#应用上下文管理器
with make_resource() as r:
    r.query()
