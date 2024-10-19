{
  description = "Quarto Devshell";

  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems =
        function:
        nixpkgs.lib.genAttrs [
          "x86_64-linux"
          "aarch64-darwin"
        ] (system: function nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          buildInputs = [
            pkgs.quarto
            pkgs.texliveFull
          ];
        };
        tex = pkgs.mkShell {
          buildInputs = [
            pkgs.texliveFull
          ];
        };
      });
    };
}
