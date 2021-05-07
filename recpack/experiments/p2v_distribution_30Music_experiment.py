# %%
from recpack.data.datasets import ThirtyMusicSessions
from recpack.algorithms.p2v_distribution_sampling import Prod2VecDistributionSampling
from recpack.splitters.scenarios import NextItemPrediction
from recpack.metrics.precision import PrecisionK
from recpack.data.matrix import to_csr_matrix

sessions_dataset = ThirtyMusicSessions("../data/datasets/30Music_sessions_3M_users_unfiltered.csv")
im = sessions_dataset.load_interaction_matrix()
# Scenario
scenario = NextItemPrediction(validation=True)
scenario.split(im)
train = scenario.train_X
test_data_in = scenario.test_data_in
test_data_out = scenario.test_data_out

val_data_in = scenario._validation_data_in
val_data_out = scenario._validation_data_out
val_data_in = to_csr_matrix(val_data_in)
val_data_out = to_csr_matrix(val_data_out)

# Initialize the model
prod2vec = Prod2VecDistributionSampling(embedding_size=50, num_neg_samples=5, window_size=2, learning_rate=0.001, stopping_criterion="precision", batch_size=500, max_epochs=10, min_improvement=0.00001, keep_last=True)

# Fit the model on training data
prod2vec.fit(train, (val_data_in, val_data_out))

test_data_in = to_csr_matrix(test_data_in)
test_data_out = to_csr_matrix(test_data_out)

# Generate predictions
predictions = prod2vec.predict(test_data_in)

# Save the model for further usage later
prod2vec.save()

# Calculate Precision@K: two embeddings
preck = PrecisionK(10)
preck.calculate(test_data_out, predictions)
sum = preck.scores_.sum()
score = sum / test_data_in.shape[0]
print(f"Score two embeddings: {score}")