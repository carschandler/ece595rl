{
  description = "Quarto Devshell";

  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.aarch64-darwin;
    in
    {
      devShells.aarch64-darwin.default = pkgs.mkShell {
        buildInputs = [
          pkgs.quarto
        ];
      };

      formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.nixfmt-rfc-style;
    };
}
