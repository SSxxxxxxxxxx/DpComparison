import os


class TrainingContext:
    def __init__(self,path = None,keepCWD=False,nextCWD=None,info = None):
        self.path=path
        self.oldCWD = None
        self.keepCWD= keepCWD
        self.nextCWD = nextCWD
        self.info = info

    def __enter__(self):
        print('Setup Environment')
        import tensorflow as tf
        import numpy as np
        import random, time, os, string, sklearn
        from evaluatingDPML.core.classifier import CHECKPOINT_DIR
        from evaluatingDPML import chdir_to_evaluating

        # Save old CWD
        self.oldCWD = os.getcwd()

        # Adust CWD
        chdir_to_evaluating()

        # Setup Checkpoint dir
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        random.seed(time.time() * random.random())
        self.path = os.path.join(CHECKPOINT_DIR, ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(20)))

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        os.environ['CHECKPOINT_DIR'] = self.path

        if self.info is not None:
            with open(os.path.join(self.path,'info.txt'),'w') as f:
                f.write(self.info)

        # Seeding
        random.seed(0)
        tf.random.set_seed(1)
        sklearn.random.seed(2)
        np.random.seed(3)


        # Set Logging
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.autograph.set_verbosity(2)
        tf.get_logger().setLevel('ERROR')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil
        # Remove Temp-dir
        shutil.rmtree(self.path)

        # Reinstate CWD
        if self.nextCWD is not None:
            os.chdir(self.nextCWD)
        elif self.keepCWD:
            os.chdir(self.oldCWD)