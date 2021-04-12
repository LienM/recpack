# %%
from recpack.data.datasets import ThirtyMusicSessionsSmall
from recpack.algorithms.p2v import Prod2Vec
from recpack.splitters.scenarios import NextItemPrediction
from recpack.metrics.precision import PrecisionK
from recpack.data.matrix import to_csr_matrix

sessions_dataset = ThirtyMusicSessionsSmall("")
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
prod2vec = Prod2Vec(embedding_size=50, negative_samples=5, window_size=2, stopping_criterion="averaged_precision", batch_size=500, max_epochs=3, prints_every_epoch=1, min_improvement=0.00001)
# Fit the model on training data
prod2vec.fit(train, (val_data_in, val_data_out))

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