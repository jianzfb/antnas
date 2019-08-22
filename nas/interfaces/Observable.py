class Event(object):
    pass

callbacks=[]


class Observable(object):
    def __init__(self, *args, **kwargs):
        super(Observable, self).__init__(*args, **kwargs)

    def subscribe(self, callback):
        global callbacks
        callbacks.append(callback)

    def fire(self, **attrs):
        e = Event()
        e.source = self
        global callbacks
        for k, v in attrs.items():
            setattr(e, k, v)
        for fn in callbacks:
            fn(e)
