// src/pow.rs
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ChatChallenge {
    pub challenge: InnerChallenge,
}

#[derive(Deserialize)]
pub struct InnerChallenge {
    pub algorithm: String,
    pub challenge: String,
    pub salt: String,
    pub signature: String,
    pub difficulty: u64,
    pub expire_at: u64,
    #[allow(dead_code)] // part of the challenge wire format; not needed to solve
    pub expire_after: u64,
    pub target_path: String,
}

#[derive(Serialize)]
pub struct SolvedChallenge {
    pub algorithm: String,
    pub challenge: String,
    pub salt: String,
    pub answer: u64,
    pub signature: String,
    pub target_path: String,
}

impl ChatChallenge {
    pub fn solve(&self) -> Option<u64> {
        self.challenge.solve()
    }
    pub fn to_solved(&self, answer: u64) -> SolvedChallenge {
        let c = &self.challenge;
        SolvedChallenge {
            algorithm: c.algorithm.clone(),
            challenge: c.challenge.clone(),
            salt: c.salt.clone(),
            answer,
            signature: c.signature.clone(),
            target_path: c.target_path.clone(),
        }
    }
}

impl InnerChallenge {
    pub fn solve(&self) -> Option<u64> {
        if self.algorithm != "DeepSeekHashV1" {
            return None;
        }
        let expire = self.expire_at;
        let salt = self.salt.clone();
        let target = self.challenge.clone();
        (0..self.difficulty)
            .into_par_iter()
            .find_any(move |&guess| {
                let text = format!("{}_{}_{}", salt, expire, guess);
                hex::encode(deepseek_hash_v1(text.as_bytes())) == target
            })
    }
}

// Standard Keccak-f[1600] round constants (24 rounds).
// DeepSeekHashV1 runs rounds 1..23, so we skip round 0.
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

const PI: [usize; 25] = [
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
];

#[inline]
fn rotl64(x: u64, n: u32) -> u64 {
    if n == 0 {
        x
    } else {
        x.rotate_left(n)
    }
}

fn keccak_f_1600(state: &mut [u64; 25]) {
    let mut c = [0u64; 5];
    let mut d = [0u64; 5];
    // Rounds 1..23 (skip round 0)
    for round in RC.iter().skip(1) {
        // θ
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        }
        for x in 0..25 {
            state[x] ^= d[x % 5];
        }
        // ρ and π
        let old = *state;
        for i in 0..25 {
            state[PI[i]] = rotl64(old[i], RHO[i]);
        }
        // χ
        for row in 0..5 {
            let b = row * 5;
            let t0 = state[b];
            let t1 = state[b + 1];
            let t2 = state[b + 2];
            let t3 = state[b + 3];
            let t4 = state[b + 4];
            state[b] ^= (!t1) & t2;
            state[b + 1] ^= (!t2) & t3;
            state[b + 2] ^= (!t3) & t4;
            state[b + 3] ^= (!t4) & t0;
            state[b + 4] ^= (!t0) & t1;
        }
        // ι
        state[0] ^= round;
    }
}

pub fn deepseek_hash_v1(input: &[u8]) -> [u8; 32] {
    const RATE: usize = 136;
    assert!(input.len() < RATE - 1, "PoW input must be single-block");

    let mut state = [0u64; 25];
    let mut block = [0u8; RATE];

    // SHA3 padding: 0x06 ... 0x80
    block[..input.len()].copy_from_slice(input);
    block[input.len()] = 0x06;
    block[RATE - 1] |= 0x80;

    // Absorb: standard little-endian lane loading
    for i in 0..(RATE / 8) {
        let lane = u64::from_le_bytes([
            block[i * 8],
            block[i * 8 + 1],
            block[i * 8 + 2],
            block[i * 8 + 3],
            block[i * 8 + 4],
            block[i * 8 + 5],
            block[i * 8 + 6],
            block[i * 8 + 7],
        ]);
        state[i] ^= lane;
    }

    keccak_f_1600(&mut state);

    // Squeeze: first 32 bytes, standard little-endian lane output
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&state[i].to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        // From Deepseek-Harness test suite
        let salt = "523e5a648e576f873cdd";
        let expire_at = 1_782_779_621_948u64;
        let answer = 100_080u64;
        let expected = "28454c8523234d6a370a3d89d1081fc93904129730835dbb5eb6a66f23fd5b39";

        let text = format!("{}_{}_{}", salt, expire_at, answer);
        let hash = hex::encode(deepseek_hash_v1(text.as_bytes()));
        assert_eq!(hash, expected);
    }
}
