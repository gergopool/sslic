from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 global_step=0, batch_size=1, world_size=1, log_per_sample=1e4):
        self.writer = SummaryWriter(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.global_step = global_step
        self.warned_missing_grad = False
        self.batch_size = batch_size
        self.world_size = world_size
        self.dist_bs = self.batch_size * self.world_size
        self.log_per_sample = log_per_sample

    def add_scalar(self, tag, scalar_value, walltime=None, new_style=False, double_precision=False, force=False):
        if self.need_log() or force:
            self.writer.add_scalar(tag, scalar_value, self.global_step, walltime, new_style, double_precision)

    def add_histogram(self, tag, values, bins='tensorflow', walltime=None, max_bins=None, force=False):
        if self.need_log() or force:
            self.writer.add_histogram(tag, values, self.global_step, bins, walltime, max_bins)

    def add_image(self, tag, img_tensor, walltime=None, dataformats='CHW', force=False):
        if self.need_log() or force:
            self.writer.add_image(tag, img_tensor, self.global_step, walltime, dataformats)

    def add_figure(self, tag, figure, close=True, walltime=None, force=False):
        if self.need_log() or force:
            self.writer.add_figure(tag, figure, self.global_step, close, walltime)

    def add_text(self, tag, text_string, walltime=None, force=False):
        if self.need_log() or force:
            self.writer.add_text(tag, text_string, self.global_step, walltime)

    def log_describe(self, name, tensor, force=False):
        if self.need_log() or force:
            self.add_scalar(f"{name}_mean", tensor.mean())
            self.add_scalar(f"{name}_std", tensor.std())
            self.add_histogram(f"{name}", tensor)

    def log_config(self, args):
        config_text = ""
        for name, value in args.items():
            config_text += f"--{name} : {value}  \n"
        self.writer.add_text("config", config_text)

    def step(self):
        self.global_step += self.dist_bs

    def need_log(self):
        assert self.log_per_sample >= self.dist_bs, \
            f"per_step ({self.log_per_sample}) < distributed batch size ({self.dist_bs})"
        return (round(self.global_step / self.dist_bs) % round(self.log_per_sample / self.dist_bs)) == 0

class EmptyLogger(Logger):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self
