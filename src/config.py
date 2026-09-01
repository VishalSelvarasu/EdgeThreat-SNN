RANDOM_STATE = 42
EPOCHS = 20
LEARNING_RATE = 1e-3
TIME_STEPS = 25

FEATURE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count", "logged_in"
]

LABEL_COLUMN = "label"
