from utils.logger import get_logger
import tensorflow as tf
from utils.train_val_function import train_step, val_step

logger = get_logger(__name__)

def train_model():
    # Setup TensorFlow to use GPU 0
    logger.info(f"Start --> Configuring the GPU for Training")
    try:
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            tf.config.set_visible_devices(tf_gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(tf_gpus[0], True)
            STUDENT_DEVICE = '/GPU:0'  # TensorFlow device string
            logger.info(f"✅ TensorFlow assigned to GPU 0: {STUDENT_DEVICE}")
        else:
            STUDENT_DEVICE = '/CPU:0'
            logger.info("⚠️ No GPU found for TensorFlow. Using CPU.")
    except Exception as e:
        STUDENT_DEVICE = '/CPU:0'
        logger.info
        (f"❌ TensorFlow device setup failed: {e}. Using CPU.")
    logger.info(f"Done --> Configuring the GPU for Training")


from tqdm import tqdm
def start_train(
        EPOCHS,
        train_metric,
        val_metric,
        train_ds,
        LOG_INTERVAL,
        optimizer,
        checkpoint_manager_best,
        checkpoint_manager_last,
        val_ds,
        STUDENT_MODEL,
        train_step,
        val_step
):
    
    current_best_loss = float('inf')
    #################### TRAINING LOOP #################### #
    for epoch in range(EPOCHS):
        train_metric.reset_state() # reset train metric
        val_metric.reset_state()  # reset val metric 
        

        # tqdm for progress bar
        pbar = tqdm(train_ds, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for i, (name, X, Y) in enumerate(pbar):
            # X (16, 256, 256, 3) Y (16, 256, 256, 1)
            total_loss = train_step(
                images=X,
                true_depth=Y,
                STUDENT_MODEL = STUDENT_MODEL,
                optimizer = optimizer,
                train_metric = train_metric
            )

            pbar.set_postfix({
                "total_loss": float(total_loss),
            })

            current_loss = train_metric.result().numpy()
            # Log every few steps
            if i % LOG_INTERVAL == 0:
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                logger.info(f"Step {i}: Train Loss = {total_loss}")

        #################### VALIDATION STEP #################### #
        pbar_val = tqdm(val_ds, desc=f"Validation Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for j, (name, X, Y) in enumerate(pbar_val):
            val_loss = val_step(
                images=X,
                true_depth=Y,
                STUDENT_MODEL = STUDENT_MODEL, 
                val_metric = val_metric
            )

            pbar_val.set_postfix({
                "val_loss": float(val_loss)
            })

        #################### SAVE CHECKPOINTS #################### #
        current_loss = train_metric.result().numpy()
        if current_loss < current_best_loss:
            checkpoint_path_best_model = checkpoint_manager_best.save()
            print(f"[INFO] New Best Model found with error: {current_loss:.4f} | Saved at: {checkpoint_path_best_model}")
            current_best_loss = current_loss

        checkpoint_save_path = checkpoint_manager_last.save()
        print(f"[INFO] Last checkpoint saved at: {checkpoint_save_path}")

        #################### LOG EPOCH STATS #################### #
        current_lr = optimizer.learning_rate(epoch).numpy() if callable(optimizer.learning_rate) else optimizer.learning_rate.numpy()
        print(f"[INFO] Epoch {epoch+1} | LR = {current_lr:.6f}")
        print(f"Epoch {epoch+1}: Avg Train Loss = {train_metric.result():.4f} | Avg Val Loss = {val_metric.result():.4f}")
    return STUDENT_MODEL
    
