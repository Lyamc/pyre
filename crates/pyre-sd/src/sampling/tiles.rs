//! Ported from cabin (cabin-nodes/src/tiles.rs).
//! Tiled decoding: split a latent into overlapping tiles, decode each, and
//! blend the results with linear feathering over the overlaps. Pure; the VAE
//! Decode node does the device work.

/// Latent tile size (512 px) and minimum overlap (128 px) used by the VAE
/// Decode node.
pub const TILE: usize = 64;
pub const OVERLAP: usize = 16;

/// Start offsets of tiles of size `tile` covering `0..len`, neighbours
/// overlapping by at least `overlap`, spread evenly, first at 0 and last
/// ending at `len`. One tile if it fits.
pub fn starts(len: usize, tile: usize, overlap: usize) -> Vec<usize> {
    assert!(overlap < tile, "overlap must be smaller than the tile");
    if len <= tile {
        return vec![0];
    }
    let n = (len - overlap).div_ceil(tile - overlap);
    let span = len - tile;
    (0..n).map(|i| (i * span + (n - 1) / 2) / (n - 1)).collect()
}

/// A tile in latent coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Tiles covering an `h` x `w` latent.
pub fn plan(h: usize, w: usize, tile: usize, overlap: usize) -> Vec<Tile> {
    let (th, tw) = (tile.min(h), tile.min(w));
    starts(h, tile, overlap)
        .into_iter()
        .flat_map(|y| {
            starts(w, tile, overlap)
                .into_iter()
                .map(move |x| Tile { x, y, w: tw, h: th })
        })
        .collect()
}

/// Feathering weight of position `i` in a tile of `size` pixels: ramps up
/// over `ramp` pixels from each edge that borders another tile (`fade_start`,
/// `fade_end`), 1 elsewhere. Always > 0.
pub fn weight(i: usize, size: usize, ramp: usize, fade_start: bool, fade_end: bool) -> f32 {
    let ramp = ramp.max(1) as f32;
    let up = if fade_start {
        ((i as f32 + 0.5) / ramp).min(1.0)
    } else {
        1.0
    };
    let down = if fade_end {
        (((size - i) as f32 - 0.5) / ramp).min(1.0)
    } else {
        1.0
    };
    up.min(down)
}

/// Weighted sum of planar tiles over a `width` x `height` image or latent.
pub struct Blend {
    width: usize,
    height: usize,
    channels: usize,
    acc: Vec<f32>,
    total: Vec<f32>,
}

impl Blend {
    /// A blend over 3-channel planar pixels, for decoding.
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_channels(width, height, 3)
    }

    /// A blend over `channels` planes: 3 for pixels, or a latent's channel
    /// count when tiling an encode.
    pub fn with_channels(width: usize, height: usize, channels: usize) -> Self {
        Blend {
            width,
            height,
            channels,
            acc: vec![0.0; channels * width * height],
            total: vec![0.0; width * height],
        }
    }

    /// Add a planar tile (`channels * tw * th` values) at (`x0`, `y0`),
    /// feathered over `ramp` elements on edges that are inside the image.
    pub fn add(&mut self, x0: usize, y0: usize, tw: usize, th: usize, data: &[f32], ramp: usize) {
        assert_eq!(data.len(), self.channels * tw * th, "tile size");
        assert!(
            x0 + tw <= self.width && y0 + th <= self.height,
            "tile inside the image"
        );
        let (fl, fr) = (x0 > 0, x0 + tw < self.width);
        let (ft, fb) = (y0 > 0, y0 + th < self.height);
        let plane = self.width * self.height;
        let tile_plane = tw * th;
        for ty in 0..th {
            let wy = weight(ty, th, ramp, ft, fb);
            for tx in 0..tw {
                let w = wy * weight(tx, tw, ramp, fl, fr);
                let p = (y0 + ty) * self.width + x0 + tx;
                self.total[p] += w;
                let at = ty * tw + tx;
                for c in 0..self.channels {
                    self.acc[c * plane + p] += w * data[c * tile_plane + at];
                }
            }
        }
    }

    /// The blended planes (uncovered pixels are 0).
    pub fn finish(self) -> Vec<f32> {
        let plane = self.width * self.height;
        let channels = self.channels;
        let mut out = self.acc;
        for (i, v) in out.iter_mut().enumerate() {
            let t = self.total[i % plane];
            if t > 0.0 {
                *v /= t;
            }
        }
        debug_assert_eq!(out.len(), channels * plane);
        out
    }
}

