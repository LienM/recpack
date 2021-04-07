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
test = to_csr_matrix(train).toarray()
test_data_in = scenario.test_data_in
test_data_out = scenario.test_data_out

val_data_in = scenario._validation_data_in
val_data_out = scenario._validation_data_out
# Specify how much validation data you really want to use:
val_data_in = to_csr_matrix(val_data_in)[:1000, :]
val_data_out = to_csr_matrix(val_data_out)[:1000, :]

# Initialize the model
prod2vec = Prod2Vec(vector_size=50, item_vocabulary=im.shape[1])
# Fit the model on training data
embedding = prod2vec.fit(train, learning_rate=0.01, num_epochs=1, validation_in=val_data_in, validation_out=val_data_out, negative_samples=5, window=2, prints_every_epoch=1, batch=500)

test_data_in = to_csr_matrix(test_data_in)
test_data_out = to_csr_matrix(test_data_out)

# Generate predictions
predictions = prod2vec.predict(10, test_data_in, embedding)

# Save the model for further usage later
prod2vec.save()

# Calculate Precision@K
preck = PrecisionK(10)
preck.calculate(test_data_out, predictions)
sum = preck.scores_.sum()
score = sum / test_data_in.shape[0]
print(f"Score: {score}")