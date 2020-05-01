import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class MaxPairwiseDistance {

    private static final long SEED = 1234004;

    public static void main(String[] args) {

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: file_path k_coreset");
        }

        // Read points from a file whose name is provided as args[0]
        String filename = args[0];
        ArrayList<Vector> inputPoints = new ArrayList<>();
        try {
            inputPoints = readVectorsSeq(filename);
        } catch (IOException e) {
            System.out.println("Failed to read input file: " + e.getMessage());
        }

        // The size of the coreset k is given in args[1]
        int kCoreset = Integer.parseInt(args[1]);

        long startTime, endTime, duration;
        double maxDistance;

        // Exact algorithm
        startTime = System.currentTimeMillis();
        maxDistance = exactMPD(inputPoints);
        endTime = System.currentTimeMillis();
        duration = endTime - startTime;

        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + duration + " ms");
        System.out.println();

        // Two approximation algorithm
        startTime = System.currentTimeMillis();
        maxDistance = twoApproxMPD(inputPoints, kCoreset);
        endTime = System.currentTimeMillis();
        duration = endTime - startTime;

        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("k = " + kCoreset);
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + duration + " ms");
        System.out.println();

        // Farthest-First Traversal k-center based algorithm
        startTime = System.currentTimeMillis();
        ArrayList<Vector> kCenters = kCenterMPD(inputPoints, kCoreset);
        maxDistance = exactMPD(kCenters);
        endTime = System.currentTimeMillis();
        duration = endTime - startTime;

        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("k = " + kCoreset);
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + duration + " ms");
    }


    /**
     * Finds the maximum distance between any pair of points in the dataset given as parameter
     * @param inputPoints ArrayList of 2D points represented as Vector
     * @return Maximum L2 euclidean distance between any pair of points
     */
    public static double exactMPD(ArrayList<Vector> inputPoints) {
        int size = inputPoints.size();
        double maxDistance = 0;
        for (int i=0; i<size; i++) {
            for (int j=i+1; j<size; j++) {
                // working with squared distance for efficiency
                double dist = Vectors.sqdist(inputPoints.get(i), inputPoints.get(j));
                if (dist > maxDistance)
                    maxDistance = dist;
            }
        }
        return Math.sqrt(maxDistance);
    }

    /**
     * Finds a two-approximation of the maximum pairwise in the dataset given as parameter
     * @param inputPoints ArrayList of 2D points represented as Vector
     * @param k Size of the coreset, larger k gives better approximation but longer computation.
     * @return Approximation of maximum L2 euclidean distance between any pair of points
     */
    public static double twoApproxMPD(ArrayList<Vector> inputPoints, int k) throws IllegalArgumentException {

        List<Vector> selectedPoints = pickNDistinctRandomElements(inputPoints, k);

        // find maximum distance d(x,y), over all x in selected points and y in input points
        double maxDistance = 0;
        for (Vector selectedPoint : selectedPoints) {
            for (Vector inputPoint : inputPoints) {
                double dist = Vectors.sqdist(selectedPoint, inputPoint);
                if (dist > maxDistance)
                    maxDistance = dist;
            }
        }
        return Math.sqrt(maxDistance);
    }

    /**
     * Finds k centers using Farthest-First Traversal algorithm
     * @param inputPoints ArrayList of 2D points represented as Vector
     * @param k Size of the coreset i.e. the number of centers to pick from the input points.
     * @return ArrayList of Vector representing the k centers
     */
    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> inputPoints, int k) {
        int size = inputPoints.size();
        ArrayList<Vector> selectedPoints = new ArrayList<>();

        // this array holds the min distance of each point to the selected centers
        double[] minDistance = new double[size];

        // select first point as a center
        int selected = 0;
        selectedPoints.add(inputPoints.get(selected));

        // initialize the array of min distances as the distance of each point to the first selected center
        for (int i=0; i<size; i++)
            minDistance[i] = Vectors.sqdist(inputPoints.get(selected), inputPoints.get(i));

        // selection of k centers using farthest first traversal algorithm
        for (int i=1; i<k; i++) {
            // select point farthest from the selected centers
            selected = 0;
            for (int j=0; j<size; j++) {
                if (minDistance[j] > minDistance[selected])
                    selected = j;
            }
            selectedPoints.add(inputPoints.get(selected));

            // update array of distances of the points to the selected centers
            for (int j=0; j<size; j++) {
                double dist = Vectors.sqdist(inputPoints.get(selected), inputPoints.get(j));
                if (dist < minDistance[j])
                    minDistance[j] = dist;
            }
        }

        return selectedPoints;
    }

    /**
     * Select n distinct random elements from a list
     * @param list List from which pick the elements
     * @param n Number of elements to pick
     * @param <E> Type of elements in the list
     * @return n distinct elements from the list picked at random
     */
    private static <E> List<E> pickNDistinctRandomElements(List<E> list, int n) {
        Random randomGenerator = new Random(SEED);
        int length = list.size();

        if (n > length)
            throw new IllegalArgumentException("Assertion: trying to pick more elements than list size");

        for (int i = length - 1; i >= length - n; --i) {
            Collections.swap(list, i , randomGenerator.nextInt(i + 1));
        }
        return list.subList(length - n, length);
    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods to read input from file
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(MaxPairwiseDistance::strToVector)
                .forEach(result::add);
        return result;
    }
}
