from tensorflow.keras.models import load_model 

model = load_model("supercombo.keras")
print(model.layers[0].input_shape)