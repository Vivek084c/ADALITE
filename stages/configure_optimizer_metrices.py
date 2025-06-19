import tensorflow as tf
import os
from utils.logger import get_logger

logger = get_logger(__name__)

# Custom Cyclical Learning Rate Scheduler
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_lr, step_size):
        super().__init__()
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def __call__(self, step):
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        scale = tf.maximum(0., 1 - x)
        lr = self.initial_lr + (self.max_lr - self.initial_lr) * scale
        return lr

def func_configure_optimizer_metrics_checkpoints(TOTAL_IMGS, BATCH_SIZE, STUDENT_MODEL, CHECKPOINT_DIR):
    # Define your values
    logger.info(f"Start --> setup for optimizer, lr_schduler and chekcpoints")
    initial_lr = 1e-5
    max_lr = 3e-4
    steps_per_epoch = TOTAL_IMGS//BATCH_SIZE
    step_size = 2 * steps_per_epoch  # one cycle every 2 epochs

    # Instantiate the scheduler
    lr_schedule = CyclicalLearningRate(
        initial_lr=initial_lr,
        max_lr=max_lr,
        step_size=step_size
    )

    # Instantiate the AdamW optimizer with scheduler
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    train_metric = tf.keras.metrics.Mean(name="train_loss")
    val_metric = tf.keras.metrics.Mean(name="val_loss")


    # checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    # Shared checkpoint object
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=STUDENT_MODEL)

    # Manager for saving last 10 epochs
    checkpoint_manager_last = tf.train.CheckpointManager(
        checkpoint, directory=os.path.join(CHECKPOINT_DIR, "last"), max_to_keep=5
    )

    # Manager for best model
    checkpoint_manager_best = tf.train.CheckpointManager(
        checkpoint, directory=os.path.join(CHECKPOINT_DIR, "best"), max_to_keep=2
    )

    logger.info(f"Done --> setup for optimizer, lr_schduler and chekcpoints")

    return lr_schedule, optimizer, train_metric, val_metric, checkpoint_manager_last, checkpoint_manager_best