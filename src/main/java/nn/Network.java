package nn;

import java.util.Arrays;

public class Network {

    private double[][] output;      //[layer][neuron]
    private double[][][] weigths;   //[layer][neuron][previous neuron]
    private double[][] bias;        //[layer][neuron]

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

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = new double[NETWORK_LAYER_SIZES[i]];

            if(i > 0) {
                this.weigths[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i - 1]];
            }
        }
    }

    public static void main(String[] args) {
        Network network = new Network(4,1,3,4);
        System.out.println(Arrays.toString(network.calculate(0.2, 0.9, 0.3, 0.4)));
    }

    public double[] calculate(double... input) {
        if(input.length != this.INPUT_SIZE) return null;
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double summ = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    summ += output[layer - 1][prevNeuron] * weigths[layer][neuron][prevNeuron];
                }

                output[layer][neuron] = sigmoid(summ);
            }
        }
        return output[NETWORK_SIZE - 1];
    }

    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }
}
