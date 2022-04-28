class MultiCropTransform:

    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, x):
        return [trans(x) for trans in self.trans_list]