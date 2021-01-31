import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;


public class MaxPairwiseDist {

    private final static Random RAND = new Random(123);

    public static void main(String... args) {
        if (args.length != 2)
            throw new IllegalArgumentException("USAGE: filepath k");

        String filename = args[0];
        ArrayList<Vector> inputPoints = new ArrayList<>();
        try {
            inputPoints = readVectorsSeq(filename);
        } catch (IOException e) {
            System.out.println("Failed to read input file: " + e.getMessage());
        }

        int k = Integer.parseInt(args[1]);

        long startMillis, endMillis;
        double maxDistance = 0;

        startMillis = System.currentTimeMillis();
        maxDistance = exactMPD(inputPoints);
        endMillis = System.currentTimeMillis();
        System.out.println(
                        "\n\nEXACT ALGORITHM" +
                        "\nMax distance = " + maxDistance +
                        "\nRunning time = " + (endMillis - startMillis)
        );

        startMillis = System.currentTimeMillis();
        maxDistance = twoApproxMPD(inputPoints, k);
        endMillis = System.currentTimeMillis();
        System.out.println(
                        "\n\n2-APPROXIMATION ALGORITHM" +
                        "\nk = " + k +
                        "\nMax distance = " + maxDistance +
                        "\nRunning time = " + (endMillis - startMillis)
        );

        startMillis = System.currentTimeMillis();
        maxDistance = exactMPD(kCenterMPD(inputPoints, k));
        endMillis = System.currentTimeMillis();
        System.out.println(
                        "\n\nk-CENTER-BASED ALGORITHM" +
                        "\nk = " + k + "\nMax distance = " + maxDistance +
                        "\nRunning time = " + (endMillis - startMillis)
        );
    }

    public static double exactMPD(ArrayList<Vector> S) {
        double maxDist = 0;
        for (int i=0; i<S.size(); i++)
            for (int j=i+1; j<S.size(); j++) {
                double currDist = Vectors.sqdist(S.get(j), S.get(i));
                if (currDist > maxDist)
                    maxDist = currDist;
            }
        return Math.sqrt(maxDist);
    }

    public static double twoApproxMPD(ArrayList<Vector> S, int k) {
        if (k > S.size())
            throw new IllegalArgumentException("Assertion k < |S| failed");

        List<Vector> subset = pickNDistinctRandomElements(S, k);

        double maxDist = 0;
        for (Vector from : subset)
            for (Vector to : S) {
                double currDist = Vectors.sqdist(from, to);
                if (currDist > maxDist)
                    maxDist = currDist;
            }

        return Math.sqrt(maxDist);
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) {
        if (k > S.size())
            throw new IllegalArgumentException("Assertion k < |S| failed");

        // holds selected centers
        ArrayList<Vector> C = new ArrayList<>();

        // randomly select the first center
        int lastCenter = RAND.nextInt(S.size());
        C.add(S.get(lastCenter));

        double[] pointCDistance = new double[S.size()];
        for (int i=0; i<S.size(); i++)
            pointCDistance[i] = Vectors.sqdist(S.get(lastCenter), S.get(i));

        for (int i=1; i<k; i++) {

            double currMaxDistance = -1;
            int nextCenter = -1;

            for (int j=0; j<S.size(); j++) {
                // Check if the last selected center is closer to p. If so, update closest center distance for p
                // Note that if j == lastCenter then its updatedDist is set to zero
                double updatedDist = Vectors.sqdist(S.get(j), S.get(lastCenter));
                if (updatedDist < pointCDistance[j])
                    pointCDistance[j] = updatedDist;

                // Check if p is the farthest point from C (ie if p could be the next center)
                if (pointCDistance[j]  > currMaxDistance) {
                    currMaxDistance = pointCDistance[j] ;
                    nextCenter = j;
                }
            }

            C.add(S.get(nextCenter));
            lastCenter = nextCenter;
        }
        return C;
    }

    private static <E> List<E> pickNDistinctRandomElements(List<E> list, int n) {
        int length = list.size();

        if (n > length)
            throw new IllegalArgumentException("Assertion n <= list.size() failed");
        else if (n == length)
            return list;

        for (int i = length - 1; i >= length - n; --i) {
            Collections.swap(list, i , RAND.nextInt(i + 1));
        }
        return list.subList(length - n, length);
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(MaxPairwiseDist::strToVector)
                .forEach(result::add);
        return result;
    }
}
