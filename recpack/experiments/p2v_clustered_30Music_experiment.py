# %%
from recpack.data.datasets import ThirtyMusicSessions
from recpack.algorithms.p2v_clustered import Prod2VecClustered
from recpack.splitters.scenarios import NextItemPrediction
from recpack.metrics.precision import PrecisionK
from recpack.data.matrix import to_csr_matrix

sessions_dataset = ThirtyMusicSessions("./recpack/recpack/data/datasets/30Music_sessions_3M_users_unfiltered.csv")
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
prod2vec = Prod2VecClustered(embedding_size=50, num_neg_samples=5, window_size=2, learning_rate=0.001, num_clusters=20, Kcl=3, stopping_criterion="precision", batch_size=500, max_epochs=10, min_improvement=0.00001, keep_last=True)
# Fit the model on training data
prod2vec.fit(train, (val_data_in, val_data_out))

prod2vec._create_clustered_ranking(train)

test_data_in = to_csr_matrix(test_data_in)
test_data_out = to_csr_matrix(test_data_out)

# Generate predictions
predictions = prod2vec.predict(test_data_in)

# Save the model for further usage later
prod2vec.save()

# Calculate Precision@K
preck = PrecisionK(10)
preck.calculate(test_data_out, predictions)
sum = preck.scores_.sum()
score = sum / test_data_in.shape[0]
print(f"Score: {score}")