MODEL_HELP = """Path to a pre-trained PyTorch BERT-style model compatible with the
fine-tuning task."""
SEQ_LEN_HELP = """Maximum number of tokens per sequence"""
BATCH_SIZE_HELP = """Number of sequences to process at a time"""
EPOCHS_HELP = """Number of passes to run over the training & validation sets"""
STEPS_HELP = """Number of training updates to carry out; overwrites epochs"""
RUNS_HELP = """Number of total runs to carry out, varying the random seed each time"""
GRAD_ACC_HELP = """Number of training set batches over which to add up the gradient
between each backward pass"""
EVAL_ACC_HELP = """Number of batches to accumulate on the GPU at a time in the
evaluation phase"""
LR_HELP = """Learning rate to pass to the optimiser"""
WARMUP_HELP = """Number of learning rate warmup steps"""
WEIGHT_DECAY_HELP = """Decoupled weight decay value to apply in the Adam optimiser"""
SAVE_HELP = """Value to pass to the `save_strategy` parameter of the transformers
TrainingArguments object (https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)"""


def add_common_training_arguments(parser):
    parser.add_argument("--model", type=str, help=MODEL_HELP)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--eval_acc", type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--save", type=str, choices={"no", "epoch", "steps"}, default="steps", help=SAVE_HELP)
    return parser