import tensorflow as tf
from tensorflow import keras
from .base_model import BaseModel


@keras.utils.register_keras_serializable(package="custom", name="StochasticModule")
class StochasticModule(keras.layers.Layer):
    """
    StochasticModule implementuje warstwę opisującą równanie (16) z artykułu
    "Stochastic Neural Networks for Cryptocurrency Price Prediction" (IEEE Access, 2020):

    Równanie (16) z artykułu:
        s_t = h_t + γ * ξ_t * (h_t - s_{t-1})

    gdzie:
    - h_t: tensor bieżących wartości aktywacji w kroku czasowym t,
    - s_{t-1}: tensor stochastycznych wartości aktywacji z poprzedniego kroku lub warstwy,
    - ξ_t: tensor zmiennych losowych niezależnych (IID) generowany w runtime 
           z rozkładu jednostajnego U(0, 1) o tych samych wymiarach co h_t,
    - gamma: współczynnik perturbacji, który określa stopień losowości w kroku czasowym t.

    W pracy autorzy modelują funkcję reaction(h_t, s_{t-1}) = h_t - s_{t-1} (Eq. (15)).
    """

    def __init__(self, gamma: float = 0.1, learnable_gamma: bool = False, name: str | None = None, **kwargs):
        """
        gamma: współczynnik perturbacji, który określa stopień losowości w kroku czasowym t.
        learnable_gamma: czy gamma jest uczony czy ustalany na początku.
        name: nazwa warstwy.
        **kwargs: dodatkowe argumenty.
        """
        super().__init__(name=name, **kwargs)

        self.init_gamma = float(gamma)
        self.learnable_gamma = bool(learnable_gamma)
        self.project = None
        self.gamma_var = None # Inicjalizacja w build, jeśli podano learnable_gamma

    def build(self, input_shape):
        """
        In the Keras API, we recommend creating layer weights in the build(self, inputs_shape) method of your layer. Like this:
        - https://keras.io/guides/making_new_layers_and_models_via_subclassing/

        Zadaniem metody build jest utworzenie zmiennych warstwy.
        Metoda build() pierwszy raz zostaje wywołana w chwili pierwszego użycia warstwy.
        Keras będzie wtedy znał wymiary wejść tej warstwy i przekaże te informacje metodzie build()
        - uczenie maszynowe z uzyciem scikit learn i tensorflow wydanie ii aurelien geron, strt. 391
        """
        h_shape = input_shape[0]
        rank = len(h_shape)
        units = int(h_shape[-1])

        # Model MLP pracuje na danych 2D, a model LSTM na danych 3D, więc musimy to obsłużyć.
        if rank == 3:
            self.project = keras.layers.TimeDistributed(keras.layers.Dense(units, use_bias=False))
        else:
            self.project = keras.layers.Dense(units, use_bias=False)

        if self.learnable_gamma:
            self.gamma_var = self.add_weight(
                name="gamma",
                shape=(),
                initializer=keras.initializers.Constant(self.init_gamma),
                trainable=True,
            )
        else:
            self.gamma_var = tf.constant(self.init_gamma, dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        a "call", the layer's forward pass.
        - https://keras.io/guides/making_new_layers_and_models_via_subclassing/#the-layer-class-the-combination-of-state-weights-and-some-computation

        s_t = h_t + γ * ξ_t * (h_t - s_{t-1})
        """
        h_t, s_t_minus_1 = inputs
        s_t_minus_1 = self.project(s_t_minus_1)

        # ξ_t ~ U(0,1)
        rand = tf.random.uniform(tf.shape(h_t), minval=0.0, maxval=1.0, dtype=h_t.dtype)
        gamma_value = self.gamma_var

        # The results obtained by our models on test data
        # using stochastic layers in the neural networks are presented.
        # The parameters of the trained model remain the same as in
        # the deterministic models.
        if training is not None:
            training_flag = tf.cast(training, tf.bool)
            gamma_eff = tf.where(training_flag, tf.cast(0.0, h_t.dtype), tf.cast(gamma_value, h_t.dtype))
        else:
            gamma_eff = tf.cast(gamma_value, h_t.dtype)

        reaction = h_t - s_t_minus_1  # Eq. (15)
        return h_t + gamma_eff * rand * reaction  # Eq. (16)

    def get_config(self):
        """
        If you need your custom layers to be serializable as part of a Functional model,
        you can optionally implement a get_config() method:
        - https://keras.io/guides/making_new_layers_and_models_via_subclassing/#you-can-optionally-enable-serialization-on-your-layers
        """
        base_config = super().get_config()
        base_config.update({
            "gamma": float(self.init_gamma),
            "learnable_gamma": bool(self.learnable_gamma),
        })
        return base_config
    
    def from_config(cls, config):
        config = config.copy()
        return cls(**config)


class BaseStochasticModel(BaseModel):
    """
    Baza dla modeli stochastycznych
    """

    StochasticModule = StochasticModule



