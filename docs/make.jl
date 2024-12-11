push!(LOAD_PATH,"../src/")

using BayesianFactorZoo
using Documenter

makedocs(
    sitename = "BayesianFactorZoo.jl",
    modules = [BayesianFactorZoo],
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "Utility Functions" => "api/utils.md"
        ]
    ],
    warnonly = [:missing_docs],  # Only warn about missing docs instead of erroring
    checkdocs = :exports          # Only check exported names
)

deploydocs(;
    repo = "github.com/eohne/BayesianFactorZoo.jl.git",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"]
)