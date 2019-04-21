import tensorflow as tf
import maddpg.common.ops as ops
import collections

RRLCellTuple = collections.namedtuple("RRLCellTuple", ("query", "key", "value", "message"))

class RRL(tf.nn.rnn_cell.RNNCell):

    """
    Input:
    Query: [?, q_units]
    Keys: [?, n * q_units]
    Values: [?, n * msg_units]
    Message: [?, msg_units]

    """

    def __init__(self, prevState, newQuery, newKeys, newValues, batchSize, q_units, msg_units, num_agents, reuse=None):

        self.num_agents = num_agents
        self.prevState = prevState
        self.query = newQuery
        self.key = newKeys
        self.value = newValues
        self.q_units = q_units
        self.k_units = q_units
        self.v_units = msg_units
        self.msg_units = msg_units
        self.batchsize = batchSize
        self.none = tf.zeros((batchSize, 1), dtype=tf.float32)

        # self.train = train
        self.reuse = reuse

    @property
    def state_size(self):
        return RRLCellTuple(self.q_units, self.num_agents * self.q_units, self.num_agents * self.v_units, self.msg_units)

    @property
    def output_size(self):
        return 1

    """
    Attention Unit: computes the new messsage -- the reasoning operation

    The unit is recurrent: it receives the query and its previous state

    Args:
        queryInput: external input to the attention unit (query from receiving agent)
        [batchsize, queryDim]


        prevKey: the previous key hidden state value
        [batchsize, num_agents * queryDim]

        valueBase: external input to the attention unit (values from all 'n', including self)
        [batchsize, num_agents * valDim]

        prevValue: the previous key hidden state value
        [batchsize, num_agents * valDim]
    """

    def attn(self, prevQuery, prevKey, prevValue, prevMessage, name="", reuse=None):

        # Merge previous query state and new state together
        """
        qDim = attn_units, used for linear operation
        newQuery: [None, 2*attn_units]
        qDim = 2*attn_units
        newQuery: [None, attn_units]
        """

        qDim = self.q_units
        newQuery = tf.concat([self.query, prevQuery], axis=1)
        qDim += qDim
        newQuery = ops.linear(newQuery, qDim, self.q_units, name="query_merge")

        newKeyInt = self.key * prevKey  # This is fine
        newKeyInt = tf.reshape(newKeyInt, [-1, self.num_agents, self.q_units])
        newKey = tf.reshape(self.key, [-1, self.num_agents, self.q_units])

        newKeyProj = []

        for i in range(self.num_agents):
            # Get shape (?, 32)
            mergeKey = tf.concat([newKeyInt[:, i, :], newKey[:, i, :]], axis=1)
            kDim = 2 * self.q_units
            # Get shape [(?, q_units), (?, q_units), ...]
            newKeyProj.append(ops.linear(mergeKey, kDim, self.q_units, name="key_interact"+str(i)))

        """
        Prepare inputs for batch dot, by reshaping to get a matrix for each agent
        Old Dimension: [None, num_agents * keyDim]
        New Dimension: [None, num_agents, keyDim]        
        """
        # Stack all of them together to get (?, num_agents, q_units)
        newKeyProj = tf.stack(newKeyProj, axis=1)
        newAttn = tf.einsum('bi,bji->bj', newQuery, newKeyProj)
        newAttn = tf.nn.softmax(newAttn)

        prevValue = tf.reshape(prevValue, [-1, self.num_agents, self.v_units])
        newValue = tf.reshape(self.value, [-1, self.num_agents, self.v_units])
        newValProj = []

        for i in range(self.num_agents):
            # print("New error prevValue", prevValue[:, 1, :])
            # print("New Value error", newValue[:, 1, :])
            mergeVal = tf.concat([prevValue[:, i, :], newValue[:, i, :]], axis=1)
            # print("Merged value", mergeVal)
            vDim = 2 * self.v_units
            newValProj.append(ops.linear(mergeVal, vDim, self.v_units, name="value_interact" + str(i)))

        newValProj = tf.stack(newValProj, axis=1)
        # print(newValProj, "value that is being saved in memory")
        newValue = tf.einsum('bi,bij->bj', newAttn, newValProj)
        # print("Final New Message", newValue)

        message = newValue * prevMessage
        message = tf.concat([message, newValue], axis=1)
        msgDim = 2 * self.msg_units
        message = ops.linear(message, msgDim, self.msg_units, name="message_interact")

        """
        Prepare inputs for batch dot, by reshaping to get a matrix for each agent
        Old Dimension: [None, num_agents * valDim]

        New Dimension: [None, num_agents, valDim]        
        """

        z = tf.sigmoid(ops.linear(newQuery, self.q_units, self.msg_units, name="gate", bias=True))
        newMessage = message * z + message * (1 - z)

        #Prepare values to be stored in the tuple
        newKeyProj = tf.layers.flatten(newKeyProj)
        newValProj = tf.layers.flatten(newValProj)

        """Dimension that are saved and require feeding too
        Query: [?, q_units]
        Key: [?, n * q_units]
        Value: [?, n * msg_units]
        Message: [?, msg_units]
        """
        return newQuery, newKeyProj, newValProj, newMessage

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse=self.reuse):

            stateQuery = state.query
            stateKey = state.key
            stateValue = state.value
            stateMessage = state.message
            memQuery, memKey, memValue, memMessage = self.attn(stateQuery, stateKey, stateValue, stateMessage)
            # print("memvalue print to check", memValue)

        newState = RRLCellTuple(memQuery, memKey, memValue, memMessage)

        return self.none, newState

    def initState(self, name, dim, initType, batchSize):
        if initType == "PRM":
            prm = tf.get_variable(name, shape=(dim,),
                                  initializer=tf.random_normal_initializer())
            initState = tf.tile(tf.expand_dims(prm, axis=0), [batchSize, 1])
        elif initType == "ZERO":
            if name == "initQuery":
                initState = tf.zeros((batchSize, self.q_units), dtype=tf.float32)
            elif name == "initKey":
                initState = tf.zeros((batchSize, self.k_units), dtype=tf.float32)
            elif name == "initValue":
                initState = tf.zeros((batchSize, self.v_units), dtype=tf.float32)
            elif name == "initMessage":
                initState = tf.zeros((batchSize, self.msg_units), dtype=tf.float32)

        return initState

    def zero_state(self, batchSize, dtype=tf.float32):

        initialQuery = self.initState("initQuery", self.q_units, "ZERO", batchSize)
        initialKey = self.initState("initKey", self.k_units, "ZERO", batchSize)
        initialValue = self.initState("initValue", self.v_units, "ZERO", batchSize)
        initialMessage = self.initState("initMessage", self.msg_units, "ZERO", batchSize)

        return RRLCellTuple(initialQuery, initialKey, initialValue, initialMessage)

    def reset_rrl_state(self, batchSize, dtype=tf.float32):

        initialQuery = self.initState("initQuery", self.q_units, "ZERO", batchSize)
        initialKey = self.initState("initKey", self.k_units, "ZERO", batchSize)
        initialValue = self.initState("initValue", self.v_units, "ZERO", batchSize)
        initialMessage = self.initState("initMessage", self.msg_units, "ZERO", batchSize)

        return RRLCellTuple(initialQuery, initialKey, initialValue, initialMessage)