import numpy as np
import scipy.special
class neuralNetwork:

    def __init__(self, input_node, hidden_node, output_node, learning_rate):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.learning_rate = learning_rate

        self.action_function = lambda x:scipy.special.expit(x)

        self.wih = np.random.normal(0.0, pow(self.hidden_node, -0.5),\
            (self.hidden_node, self.input_node))
        self.who = np.random.normal(0.0, pow(self.output_node, -0.5),\
            (self.output_node, self.hidden_node))
        pass

    def train(self, input_list, target_list):
        # 输入
        inputs = np.array(input_list, ndmin=2).T
        # 目标
        targets = np.array(target_list, ndmin=2).T
        print(inputs.shape, targets.shape)
        # 输入和隐藏层运算 
        hidden_input_node = np.dot(self.wih, inputs)
        # 隐藏层输出
        hidden_output = self.action_function(hidden_input_node)
        # 隐藏层和输出层运算 
        output_hidden_node = np.dot(self.who, hidden_output)
        final_output = self.action_function(output_hidden_node)
        
        # 输出层和隐藏层
        output_error = final_output - targets
        
        print("output_error",output_error.shape)
        print("final_output",final_output.shape)
        print("target_list",targets.shape)
        print(" hidden_out.shape" ,hidden_output.shape, "error.shape" \
           ,output_error.shape, "final_output.shape" ,final_output.shape)
        self.who += self.learning_rate *\
             np.dot((output_error * final_output * (1.0 - final_output)), 
             np.transpose(hidden_output))
        # 隐藏层和输入层
        error_hidden = np.dot(self.who.T, output_error)
        self.wih += self.learning_rate *\
             np.dot((error_hidden * hidden_output * (1.0 - hidden_output)), 
             np.transpose(inputs))

        # 
        pass

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_input_node = np.dot(self.wih, inputs)
        hidden_input_node = self.action_function(hidden_input_node)
        output_hidden_node = np.dot(self.who, hidden_input_node)
        output_hidden_node = self.action_function(output_hidden_node)
        pass
        return output_hidden_node

nn = neuralNetwork(3,3,3,0.5)

b = [ 0.39728136, 0.50312606, 0.46538533]
nn.train([1,1,1],b)

