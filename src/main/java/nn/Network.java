package nn;

import java.util.Arrays;

public class Network {

    private double[][] output;              //[layer][neuron]
    private double[][][] weigths;           //[layer][neuron][previous neuron]
    private double[][] bias;                //[layer][neuron]

    private double[][] errorSignal;         //for Backpropagation train
    private double[][] outputDerivative;    //for Backpropagation train

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;


    public Network(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weigths = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.errorSignal = new double[NETWORK_SIZE][];
        this.outputDerivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = new double[NETWORK_LAYER_SIZES[i]];  //TODO fill it

            this.errorSignal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.outputDerivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            if(i > 0) {
                this.weigths[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i - 1]];   //TODO fill it
            }
        }
    }

    public static void main(String[] args) {
        Network network = new Network(4,1,3,4);
        double[] input = new double[]{0.1, 0.5, 0.2, 0.9};
        double[] target = new double[]{0, 1, 0, 0};

        for (int i = 0; i < 100000; i++) {
            network.train(input, target, 1);
        }

        double[] o = network.calculate(input);
        System.out.println(Arrays.toString(o));
    }

    public double[] calculate(double... input) {
        if(input.length != INPUT_SIZE) {
            System.out.println("CalculateError");   //TODO make Exception
            return null;
        }
        output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double summ = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    summ += output[layer - 1][prevNeuron] * weigths[layer][neuron][prevNeuron];
                }

                output[layer][neuron] = sigmoid(summ);
                outputDerivative[layer][neuron] =  output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }
        return output[NETWORK_SIZE - 1];
    }

    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    public void train(double[] input, double[] target, double eta) {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            System.out.println("Training error");     //TODO make Exception
            return;
        }
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    //see https://en.wikipedia.org/wiki/Backpropagation
    public void backpropError(double[] target) {
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            errorSignal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) 
                    * outputDerivative[NETWORK_SIZE - 1][neuron];
        }
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double summ = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    summ += weigths[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
                }
                this.errorSignal[layer][neuron] = summ * outputDerivative[layer][neuron];
            }
        }
    }

    //see https://en.wikipedia.org/wiki/Backpropagation
    public void updateWeights(double eta) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    double delta = - eta * output[layer - 1][prevNeuron] * errorSignal[layer][neuron];
                    weigths[layer][neuron][prevNeuron] += delta;
                }
                double delta = - eta * errorSignal[layer][neuron];
                bias[layer][neuron] += delta;
            }
        }
    }
}
