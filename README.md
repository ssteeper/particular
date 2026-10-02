# particular

Particles that attract each other at long range and repel at short range, plotted as colored trails.
`particles.ipynb` is the original notebook.

- `particular.py`: vectorized simulation plus named starting setups (`SCENES`). The last five (`breath`, `square`, `quartet`, `galaxy`, `constellation`) go from simplest to most complex.
- `make_gallery.py`: renders a still per scene into `gallery/`
- `search.py`: builds hundreds of random setups from seeds, simulates them in parallel, scores each one (frame fill, symmetry, smooth motion, escaping particles, clumping) and saves the best to `search/`
- `make_web.py`: builds `web/index.html`, an in-browser sandbox with every scene and the search picks, from `web/app.html`
- `make_video.py`: renders `video/particular.mp4`, which draws each scene in turn with a soundtrack generated from the simulation

```
pip install numpy matplotlib pillow
python make_gallery.py
python make_video.py video/particular.mp4   # needs ffmpeg
python search.py 800                        # about 6 minutes on 4 cores
python make_web.py
```
