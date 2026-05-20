# mutator.py
class Mutator:
    def __init__(self, cfg, masking, model, trainer=None):
        self.cfg = cfg
        self.masking = masking
        self.model = model
        self.trainer = trainer  # may be None for pure-mutation
        self._pending_outputs: list[bytearray] = []
        self._finetune_pending = False
        self._queue_get_count = 0

    def queue_get(self, filename):
        self._queue_get_count += 1
        if self._queue_get_count % self.cfg.finetune_every == 0:
            self._finetune_pending = True
        return True

    def fuzz_count(self, buf):
        self._maybe_finetune()
        tokens = self.model.tokenizer.tokenize(buf)
        if not tokens:
            self._pending_outputs = []
            return 0
        masks = [self.masking.mask(tokens) for _ in range(self.cfg.fuzz_count)]
        outputs = self.model.batch_generate(
            [mp.input_ids for mp in masks], n_samples=1,
        )
        self._pending_outputs = [
            bytearray(self.model.tokenizer.reconstruct(mp, y))
            for mp, y in zip(masks, outputs)
        ]
        return len(self._pending_outputs)

    def fuzz(self, buf, add_buf, max_size):
        return self._pending_outputs.pop(0)

    def post_run(self): pass
    def queue_new_entry(self, new, orig): pass
    def deinit(self): self.model.save_checkpoint()

    def _maybe_finetune(self):
        if not self._finetune_pending:
            return
        self._finetune_pending = False
        if self.trainer is not None:
            self.trainer.finetune()
