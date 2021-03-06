# Movie Quality Increaser


Increase the quality of your movies using a superres deep neural network inference on your GPU.

---

## Installation

For installation, you need to have openCV compiled with dnn_superres module.

```bash
git clone https://github.com/thomasarmel/movie_quality_increase.git
cd movie_quality_increase
cmake .
make
```

## Usage

### Create a temporary file with only superres video stream: 

```bash
./movie_quality_increase -f <upscale factor (2 or 4)> -i <input file path> -o <output file path> -m <models directory path>
```

**Upscale factor:** The upscale factor to use, only 2 and 4 are supported by the ESPCN model used here.

**Input file path:** The path to the input video file, must be readable by ffmpeg library.

**Output file path:** The path to the output video file, must be writable by ffmpeg library. Note that the output file will be overwritten if it already exists.

**Models directory path:** The path to the directory containing the models, provided in the repository.


### Create upscaled movie:

We will upscale per 2 the movie **old-movie.mp4** to **new-movie.mp4** using the ESPCN model.

```bash
./movie_quality_increase -f 2 -i old-movie.mp4 -o /tmp/upscale-only-video.mp4 -m ./models
ffmpeg -i old-movie.mp4 -i /tmp/upscale-only-video.mp4 -c copy -map 0:a? -map 1:v -map 0:s? -shortest new-movie.mp4
```

---

Thanks to [@fannymonori](https://github.com/fannymonori/) and
[@Saafke](https://github.com/Saafke/) for providing superres models.