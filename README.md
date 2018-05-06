# portfolio-management by ddpg

## Usage
python ddpg_model.py

## real_time plot:

![plot](https://lh3.googleusercontent.com/U9g8AOerUcFqRF8AjOg-ajTf9iCSOmLECRr5ECJSwdFdWXnIJ1koYcxs8xgOh5EhwtrQMlK8TXgOhQ)

### TODO:
- [ ] Apply fixed q target method to the model, that is, for both actor and critic model, set up two networks -- eval-net updates every step and target-net updates less times. This will fixed some trainging problems.
- [ ] Tranfer the network from keras to pytorch.
- [ ] Better actor and critic network
- [ ] LSTM to deal with time serie data
- [ ] Maybe some parallel computation, A3C (Asynchronous Advantage Actor-Critic) or maybe someting more than it.
