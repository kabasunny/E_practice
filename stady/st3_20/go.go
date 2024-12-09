package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// データセットの生成
func generateData(nSamples, nFeatures, nClusters int) *mat.Dense {
	rand.Seed(42)
	data := mat.NewDense(nSamples, nFeatures, nil)

	clusterCenters := make([][]float64, nClusters)
	for i := 0; i < nClusters; i++ {
		center := make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			center[j] = rand.Float64()*10 - 5 // -5から5の範囲の乱数
		}
		clusterCenters[i] = center
	}

	samplesPerCluster := nSamples / nClusters
	for i := 0; i < nClusters; i++ {
		for j := 0; j < samplesPerCluster; j++ {
			idx := i*samplesPerCluster + j
			for k := 0; k < nFeatures; k++ {
				data.Set(idx, k, clusterCenters[i][k]+rand.NormFloat64()) // クラスタ中心からの正規分布に従うランダム値
			}
		}
	}

	return data
}

// k-means++ アルゴリズムによる初期セントロイドの選択
func initializeCentroids(data *mat.Dense, k int) *mat.Dense {
	n, _ := data.Dims()
	probabilities := make([]float64, n)
	for i := range probabilities {
		probabilities[i] = 1.0 / float64(n)
	}

	centroids := mat.NewDense(k, 2, nil)
	distances := mat.NewDense(n, k, nil)

	for i := 0; i < k; i++ {
		idx := weightedRandomChoice(probabilities)
		centroids.SetRow(i, data.RawRowView(idx))

		for j := 0; j < n; j++ {
			distances.Set(j, i, squaredDistance(data.RawRowView(j), centroids.RawRowView(i)))
		}

		for j := 0; j < n; j++ {
			probabilities[j] = 0.0
			for l := 0; l < k; l++ {
				probabilities[j] += distances.At(j, l)
			}
		}
	}

	return centroids
}

func weightedRandomChoice(probabilities []float64) int {
	sum := 0.0
	for _, p := range probabilities {
		sum += p
	}

	r := rand.Float64() * sum
	for i, p := range probabilities {
		r -= p
		if r <= 0 {
			return i
		}
	}

	return len(probabilities) - 1
}

func squaredDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// k-means アルゴリズム
func kmeans(data *mat.Dense, k, maxIters int) (*mat.Dense, []int) {
	n, _ := data.Dims()
	centroids := initializeCentroids(data, k)
	clusters := make([]int, n)

	for iter := 0; iter < maxIters; iter++ {
		distances := mat.NewDense(n, k, nil)

		for i := 0; i < k; i++ {
			for j := 0; j < n; j++ {
				distances.Set(j, i, squaredDistance(data.RawRowView(j), centroids.RawRowView(i)))
			}
		}

		for j := 0; j < n; j++ {
			minDist := distances.At(j, 0)
			minIdx := 0
			for i := 1; i < k; i++ {
				if d := distances.At(j, i); d < minDist {
					minDist = d
					minIdx = i
				}
			}
			clusters[j] = minIdx
		}

		newCentroids := mat.NewDense(k, 2, nil)
		counts := make([]int, k)

		for j := 0; j < n; j++ {
			c := clusters[j]
			for l := 0; l < 2; l++ {
				newCentroids.Set(c, l, newCentroids.At(c, l)+data.At(j, l))
			}
			counts[c]++
		}

		for i := 0; i < k; i++ {
			if counts[i] != 0 {
				for l := 0; l < 2; l++ {
					newCentroids.Set(i, l, newCentroids.At(i, l)/float64(counts[i]))
				}
			}
		}

		if mat.EqualApprox(centroids, newCentroids, 1e-6) {
			break
		}

		centroids = newCentroids
	}

	return centroids, clusters
}

// プロット関数
func plotClusters(data *mat.Dense, centroids *mat.Dense, clusters []int) {
	p := plot.New()
	p.Title.Text = "K-means++ Clustering"
	p.X.Label.Text = "Feature 1"
	p.Y.Label.Text = "Feature 2"

	dataPoints := make(plotter.XYs, data.RawMatrix().Rows)
	for i := range dataPoints {
		dataPoints[i].X = data.RawRowView(i)[0]
		dataPoints[i].Y = data.RawRowView(i)[1]
	}

	centroidsPoints := make(plotter.XYs, centroids.RawMatrix().Rows)
	for i := range centroidsPoints {
		centroidsPoints[i].X = centroids.RawRowView(i)[0]
		centroidsPoints[i].Y = centroids.RawRowView(i)[1]
	}

	scatterData, err := plotter.NewScatter(dataPoints)
	if err != nil {
		panic(err)
	}
	scatterData.GlyphStyle.Color = plotutil.Color(2) // 濃い色に設定

	scatterCentroids, err := plotter.NewScatter(centroidsPoints)
	if err != nil {
		panic(err)
	}
	scatterCentroids.GlyphStyle.Color = plotutil.Color(0) // 濃い色に設定
	scatterCentroids.GlyphStyle.Shape = plotutil.Shape(1)

	p.Add(scatterData, scatterCentroids)
	p.Legend.Add("Data", scatterData)
	p.Legend.Add("Centroids", scatterCentroids)

	if err := p.Save(8*vg.Inch, 8*vg.Inch, "clusters.png"); err != nil {
		panic(err)
	}
}

// メイン処理
func main() {
	nSamples := 300
	nFeatures := 2
	nClusters := 3

	data := generateData(nSamples, nFeatures, nClusters)
	centroids, clusters := kmeans(data, nClusters, 100)
	plotClusters(data, centroids, clusters)
}
