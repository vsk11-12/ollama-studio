fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winres::WindowsResource::new();
        res.set_icon("app_icon.ico");
        res.set_windres_path("x86_64-w64-mingw32-windres");
        res.set_ar_path("x86_64-w64-mingw32-ar");
        if let Err(e) = res.compile() {
            eprintln!("Failed to compile Windows resource: {}", e);
            std::process::exit(1);
        }
    }
}
