fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Client-only: the gateway never serves the Inference service, it consumes it.
    tonic_build::configure()
        .build_server(false)
        .compile_protos(&["../proto/inference.proto"], &["../proto"])?;

    println!("cargo:rerun-if-changed=../proto/inference.proto");
    Ok(())
}
