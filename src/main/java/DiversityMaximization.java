import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.util.*;

public class DiversityMaximization {

    private static final long SEED = 1234004;
    private static final Random RANDOM_GENERATOR = new Random(SEED);

    public static void main(String... args) {

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path k L");
        }

        String dataFolderPath = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);

        SparkConf conf = new SparkConf(true).setAppName("Homework1 - ClassCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        long startTime, endTime, duration;

        startTime = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(dataFolderPath)
                .map(DiversityMaximization::strToVector)
                .repartition(L)
                .cache();
        long numPoints = inputPoints.count();
        endTime = System.currentTimeMillis();
        duration = endTime - startTime;

        System.out.println("Number of points = " + numPoints);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + duration);

        ArrayList<Vector> points = runMapReduce(inputPoints, k, L);

        double avgDistance = measure(points);
        System.out.println("Average distance = " + avgDistance);
    }

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        long startTime, endTime, duration;

        // ROUND 1 - Farthest first traversal on each partition
        startTime = System.currentTimeMillis();

        // Force computation
        ArrayList<Vector> points = new ArrayList<>(
            pointsRDD.mapPartitions((partition) -> {
                ArrayList<Vector> partitionList = new ArrayList<>();
                while (partition.hasNext()) {
                    partitionList.add(partition.next());
                }

                return farthestFirstTraversal(partitionList, k).iterator();
            }).collect()
        );

        endTime = System.currentTimeMillis();
        duration = endTime - startTime;
        System.out.println("Runtime of Round 1 = " + duration);

        // ROUND 2 - 2-approx algorithm
        startTime = System.currentTimeMillis();

        points = runSequential(points, k);

        endTime = System.currentTimeMillis();
        duration = endTime - startTime;
        System.out.println("Runtime of Round 2 = " + duration);

        return points;
    }

    /**
     * Finds k points using Farthest-First Traversal algorithm
     * @param inputPoints ArrayList of 2D points represented as Vector
     * @param k Size of the coreset i.e. the number of points to pick from the input points.
     * @return ArrayList of Vector representing the k points
     */
    public static ArrayList<Vector> farthestFirstTraversal(ArrayList<Vector> inputPoints, int k) {
        int size = inputPoints.size();
        ArrayList<Vector> selectedPoints = new ArrayList<>();

        // this array holds the min distance of each point to the selected centers
        double[] minDistance = new double[size];

        // select first point as a center
        int selected = RANDOM_GENERATOR.nextInt(inputPoints.size());
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
     * Computes the average distance between all pairs of points.
     * @param pointSet A set of points represented as ArrayList<Vector>
     * @return The average distance
     */
    public static double measure(ArrayList<Vector> pointSet) {
        double tot = 0;
        for (int i = 0; i < pointSet.size(); i++) {
            for (int j = i+1; j < pointSet.size(); j++)
                tot += Math.sqrt(Vectors.sqdist(pointSet.get(i), pointSet.get(j)));
        }

        return tot / (pointSet.size() * (pointSet.size() - 1) * 0.5);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    public static org.apache.spark.mllib.linalg.Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }
}
