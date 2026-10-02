# particular

Particles that attract each other at long range and repel at short range, plotted as colored trails.
`particles.ipynb` is the original notebook.

- `particular.py`: vectorized simulation plus named starting setups (`SCENES`)
- `make_gallery.py`: renders a still per scene into `gallery/`
- `make_video.py`: renders `video/particular.mp4`, which draws each scene in turn with a soundtrack generated from the simulation

```
pip install numpy matplotlib pillow
python make_gallery.py
python make_video.py video/particular.mp4   # needs ffmpeg
```
