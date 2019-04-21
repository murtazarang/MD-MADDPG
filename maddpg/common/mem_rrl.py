import tensorflow as tf
import collections
import numpy as np

import maddpg.common.ops as ops
from maddpg.common.config_args import parse_args

arglist = parse_args()

RRLCellTuple = collections.namedtuple("RRLCellTuple", ("query", "memory"))

class RRLCell(tf.nn.rnn_cell.RNNCell):
    """
    Input:
        Query: Query from the agent "i" recurrent cell
        [batchSize, query_units]

        Keys: Current representation of the observation frome each agent
        [batchSize, n, query_units]

        numAgents: number of agents
        [batchSize]

        batchSize: Tensor Scalar
        [batchSize]

        Values: Representation of observation of all agents
        [batchSize, n, mem_units]
    """

    def __init__(self, query, keys, values, numAgents, memoryDropout, readDropout, writeDropout,
                 train, batch_size, reuse=None):

        self.query = query
        self.keys = keys
        self.values = values
        self.num_agents = numAgents
        self.batch_size = batch_size

        self.dropouts = {}
        self.dropouts["memory"] = memoryDropout
        self.dropouts["read"] = readDropout
        self.dropouts["write"] = writeDropout

        self.none = tf.zeros((1, 1), dtype=tf.float32)

        self.train = train
        self.reuse = reuse

        """Cell State Size"""
        @property
        def state_size(self):
            return RRLCellTuple(arglist.query_units, arglist.mem_units)

        """Cell output size. No outputs used for now"""

        @property
        def output_size(self):
            return 1

        """
        The Control Unit
        
        Input:
        queryInput: external input to the control unit (RNN output of specific agent)
        [batchSize, query_units]
        
        Keys: Observation embeddings from all agents
        [batchSize, n, query_units]
        
        num_agents: Total number of agents in the reasoning operation
        
        query: previous query control hidden state value
        [batchSize, query_units]
        
        Returns:
        New control state
        [batchSize, query_units]
        """

    def control(self, queryInput, keys, query, contQuery=None, name="", reuse=None):
        with tf.variable_scope("control" + name, reuse=reuse):
            dim = arglist.query_units

            ## Step 1: Compute new query control state given previous control and new query.

            newContQuery = queryInput
            # print(queryInput, "input to control unit")
            if arglist.queryFeedPrev:
                newContQuery = query if arglist.queryFeedPrevAtt else contQuery
                newContQuery = tf.concat([queryInput, newContQuery], axis=-1)
                dim += arglist.query_units

                newContQuery = ops.linear(newContQuery, dim, arglist.query_units, act=arglist.queryFeedAct, name="contQuery")
                dim = arglist.query_units

            ## Step 2: Compute attention distribution over the keys and then sum them up accordingly
            # Prepare the dimensions of the control query to interact with the 'n' agent keys input
            # [num_agents, ?, query_dim]
            interactions = tf.expand_dims(newContQuery, axis=1) * keys
            # print(interactions, "inteaction with key")
            # Optional concatentation of the keys with the new interaction
            if arglist.concatKeyInteraction:
                interactions = tf.concat([interactions, keys], axis=2)
                dim += arglist.query_units
            # print(interactions, "concat with keys for proj")
            if arglist.queryProj:
                interactions = ops.linear(interactions, dim, arglist.query_units, act=arglist.queryProjAct)
                dim = arglist.query_units
            # print(interactions, "int after proj")
            # Compute the attention distribution across the query and keys and summarize for each agent?
            logits = ops.inter2logits(interactions, dim)
            # print(logits, "logits calculation")
            attention = tf.nn.softmax(ops.expMask(logits, self.num_agents))
            # print(attention, "before att2smy with keys")
            newQuery = ops.att2Smry(attention, keys)
            # print(newQuery, "Query Shape Control")

        return newQuery

    """
    The Read Unit
    
    Input:
    valueBase: [?, n, mem_size]
    memory: [?, mem_size]
    query: [?, query_units]
    
    Returns:
    Information: [?, mem_size]    
    """

    def read(self, valueBase, memory, query, name="", reuse=None):
        with tf.variable_scope("read" + name, reuse=reuse):
            dim = arglist.mem_units

            # memory dropout
            if arglist.memory_dropout:
                memory = tf.nn.dropout(memory, self.dropouts["memory"])

            ## Step 1: knowledge / memory interactions

            proj =None
            if arglist.readProjInputs:
                proj = {"dim": arglist.mem_units, "shared": arglist.readProjShared, "dropout": self.dropouts["read"]}
                dim = arglist.att_units

            # parameters for concatenating knowledge base elements
            concat = {"x": arglist.readMemConcatVB, "proj": arglist.readMemConcatProj}

            interactions, interDim = ops.mul(x=valueBase, y=memory, dim=arglist.mem_units,
                                             proj=proj, concat=concat, interMod=arglist.readMemAttType, name="memInter")


            projectedVB = proj.get("x") if proj else None #Projection value when the key 'x'= vb is added from the muls function call

            # Project memory interactions back to hidden dimension of the memory
            if arglist.readMemProj:
                interactions = ops.linear(interactions, interDim, dim, act=arglist.readMemAct)
            else:
                dim = interDim

            ## Step 2: Compute interactions with control

            if arglist.readQuery:
                # compute interactions with control
                if arglist.query_units != dim:
                    query = ops.linear(query, arglist.query_units, dim, name="queryProj")

                interactions, interDim = ops.mul(interactions, query, dim, interMod=arglist.readQueryAttType,
                                                 concat={"x": arglist.readQueryConcatInter}, name="queryIntr")

                # Optional: concatenate value base elements
                if arglist.readQueryConcatVB:
                    if arglist.readQueryConcatProj:
                        addedInp, addedDim = projectedVB, arglist.att_units


                    else:
                        addedInp, addedDim = valueBase, arglist.mem_units

                    interactions = tf.concat([interactions, addedInp], axis=-1)

                    dim += addedDim

                # optional nonlinearity
                interactions = ops.activations[arglist.readQueryAct](interactions)

            ## Step 3: Sum attentions up over the valueBase
            attention = ops.inter2logits(interactions, dim, dropout=self.dropouts["read"])

            #optionally use the projected valueBase instead of the original
            if arglist.readSmryVBProj:
                valueBase = projectedVB

            # sum up the value base according to the distribution
            information = ops.att2Smry(attention, valueBase)

            return information

    """The Write Unit
    Inputs:
        memory(values): the cell's memory state
        [batchSize, mem_units]
        
        info: the information to integrate with the memory
        [batchsize, mem_units]
        
        query: the cell's query(control) state
        [batchsize, query_units]
    
    Returns:
    new memory state
    [batchsize, mem_units]
    """

    def write(self, memory, info, query, name="", reuse=None):
        with tf.variable_scope("write" + name, reuse=reuse):
            # optionally project info
            if arglist.writeInfoProj:
                info = ops.linear(info, arglist.mem_units, arglist.mem_units, name="info")

            # optional info non-linearity
            info = ops.activations[arglist.writeInfoAct](info)

            # Get write unit inputs: previous memory, the new information
            newMemory, dim = memory, arglist.mem_units
            if arglist.writeInputs == "INFO":
                newMemory = info
            elif arglist.writeInputs == "SUM":
                newMemory += info
            elif arglist.writeInputs == "BOTH":
                newMemory, dim = ops.concat(newMemory, info, dim, mul=arglist.writeConcatMul)

            if arglist.writeMemProj or (dim != arglist.mem_units):
                newMemory = ops.linear(newMemory, dim, arglist.mem_units, name="newMemory")

            newMemory = ops.activations[arglist.writeMemAct](newMemory)

            if arglist.writeGate:
                gateDim = arglist.mem_units
                if arglist.writeGateShared:
                    gateDim = 1

                z = tf.sigmoid(ops.linear(query, arglist.query_units, gateDim, name="gate", bias=arglist.writeGateBias))
                newMemory = newMemory * z + memory * (1-z)

                if arglist.memoryBN:
                    newMemory = tf.contrib.layers.batch_norm(newMemory, decay=arglist.bnDecay, center=arglist.bnCenter,
                                                             scale=arglist.bnScale, is_training=self.train,
                                                             updates_collections=None)
        return newMemory


    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse=self.reuse):
            query = state.query
            memory = state.memory

            ## Control unit
            queryInput = ops.linear(self.query, arglist.num_units, arglist.query_units,
                                    name="queryProj")
            queryInput = ops.activations[arglist.queryInputAct](queryInput)

            newQuery = self.control(queryInput, self.keys, query, name="Control")

            ## Read Unit
            info = self.read(self.values, memory, newQuery, name="Read")

            ## Write unit
            newMemory = self.write(memory, info, newQuery, name="Write")

            newState = RRLCellTuple(newQuery, newMemory)
            return self.none, newState