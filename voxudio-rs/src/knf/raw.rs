#![allow(non_camel_case_types, dead_code)]

include!(concat!(env!("OUT_DIR"), "/knf.rs"));
pub(super) use root::{knf, std};
