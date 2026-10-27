fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // GPUI calls TaskDialogIndirect; without a manifest Windows binds comctl32 5.82 and the
    // process dies with STATUS_ENTRYPOINT_NOT_FOUND (same fix as jellyfin-gpui).
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        embed_manifest::embed_manifest(
            embed_manifest::new_manifest("Pyre.GUI")
                .dpi_awareness(embed_manifest::manifest::DpiAwareness::PerMonitorV2),
        )
        .expect("failed to embed the Windows manifest");
    }
}
