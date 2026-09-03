{
  description = "QuTE development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              uv
              python312
              pkg-config
              stdenv.cc.cc.lib
            ];

            env = {
              UV_PROJECT_ENVIRONMENT = ".venv";
              PYTHONPATH = "src:.";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
            };

            shellHook = ''
              echo "QuTE dev shell"
              echo "  setup: uv sync --extra profiling --extra qpu --group dev"
              echo "  test:  uv run pytest"
              echo "  qpu:   uv run qute-qpu-smoke --submit-qpu"
            '';
          };
        });
    };
}
