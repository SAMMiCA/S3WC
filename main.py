from __future__ import absolute_import, division, print_function

from ob_options import Options

options = Options()
opts = options.parse()
from ob_trainer import *
import time


## Main code for the experiments

# Train and validate semantic segmentation and weather classification network
# Test with various hyper-parameters according to ob_options.py

if __name__ == '__main__':
    trainer = Trainer(opts)

    if opts.test_only:
        if opts.resume is not None:
            print("semantic checkpoint found at '{}'".format(opts.resume))
        if opts.resume_SPADE is not None:
            print("SPADE checkpoint found at '{}'".format(opts.load_pretrained_SPADE_name))
        if (opts.resume is None) and (opts.resume_SPADE is None):
            raise RuntimeError("=> no checkpoint found...")

        trainer.test()
    else:
        for epoch in range(trainer.opts.start_epoch, trainer.opts.epochs):
            epoch_start_time = time.time()
            trainer.train()

            if opts.train_semantic:
                trainer.validate()
                trainer.scheduler.step()
            if opts.use_SPADE:
                # update learning rate
                trainer.validate()
                trainer.update_spade_learning_rate(epoch)

            trainer.cur_epochs += 1
            epoch_end_time = time.time()
            print("\nOne epoch Times for train_val:{}".format(epoch_end_time - epoch_start_time))

        print('=> End training\n\n')
        trainer.writer.close()
