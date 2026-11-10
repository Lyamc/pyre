fn main() {
    embed_resource::compile("tray.rc", embed_resource::NONE).manifest_optional().unwrap();
    for f in ["tray.rc", "../assets/pyre.ico", "../assets/pyre-starting.ico", "../assets/pyre-stopped.ico"] { println!("cargo:rerun-if-changed={f}"); }
}
