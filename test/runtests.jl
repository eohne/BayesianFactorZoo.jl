using Test
using Random
using Statistics
using LinearAlgebra

using BayesianFactorZoo

@testset "BayesianFactorZoo" begin

    # -----------------------------
    # Deterministic test data
    # -----------------------------
    Random.seed!(42);

    t = 60;
    k = 2;
    N = 5;

    f = randn(t, k);
    R = randn(t, N);

    f1 = f[:, 1:1];
    f2 = f[:, 2:2];

    lambda0 = zeros(k);

    # -----------------------------
    # Helper checks
    # -----------------------------
    isprobvec(x) = all(0 .<= x .<= 1);
    allfinite(x) = all(isfinite, x);

    # -----------------------------
    # TwoPassRegression
    # -----------------------------
    @testset "TwoPassRegression" begin
        res1 = TwoPassRegression(f, R);
        res2 = TwoPassRegression(f, R);

        @test hasproperty(res1, :lambda);
        @test hasproperty(res1, :lambda_gls);
        @test hasproperty(res1, :R2_adj);

        @test length(res1.lambda) == k + 1;
        @test length(res1.lambda_gls) == k + 1;

        @test allfinite(res1.lambda);
        @test allfinite(res1.lambda_gls);
        @test isfinite(res1.R2_adj);

        # Deterministic function: should reproduce exactly up to tiny fp noise
        @test isapprox(res1.lambda, res2.lambda; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.lambda_gls, res2.lambda_gls; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.R2_adj, res2.R2_adj; rtol=1e-12, atol=1e-12);
    end

    # -----------------------------
    # SDF_gmm
    # -----------------------------
    @testset "SDF_gmm" begin
        W = Matrix{Float64}(I, N + k, N + k);

        Random.seed!(123);
        res1 = SDF_gmm(R, f, W);

        Random.seed!(123);
        res2 = SDF_gmm(R, f, W);

        @test hasproperty(res1, :lambda_gmm);
        @test hasproperty(res1, :R2_adj);

        @test length(res1.lambda_gmm) == k + 1;
        @test allfinite(res1.lambda_gmm);
        @test isfinite(res1.R2_adj);

        # Deterministic conditional on inputs
        @test isapprox(res1.lambda_gmm, res2.lambda_gmm; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.R2_adj, res2.R2_adj; rtol=1e-12, atol=1e-12);
    end

    # -----------------------------
    # BayesianFM
    # -----------------------------
    @testset "BayesianFM" begin
        res1 = BayesianFM(f, R, 200; seed=123);
        res2 = BayesianFM(f, R, 200; seed=123);

        @test size(res1.lambda_ols_path) == (200, k + 1);
        @test size(res1.lambda_gls_path) == (200, k + 1);
        @test length(res1.R2_ols_path) == 200;
        @test length(res1.R2_gls_path) == 200;

        @test allfinite(res1.lambda_ols_path);
        @test allfinite(res1.lambda_gls_path);
        @test allfinite(res1.R2_ols_path);
        @test allfinite(res1.R2_gls_path);

        # Reproducibility within same Julia/version/environment
        @test isapprox(res1.lambda_ols_path, res2.lambda_ols_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.lambda_gls_path, res2.lambda_gls_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.R2_ols_path, res2.R2_ols_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.R2_gls_path, res2.R2_gls_path; rtol=1e-12, atol=1e-12);

        # Basic sanity
        m_ols = vec(mean(res1.lambda_ols_path, dims=1));
        m_gls = vec(mean(res1.lambda_gls_path, dims=1));

        @test length(m_ols) == k + 1;
        @test length(m_gls) == k + 1;
        @test allfinite(m_ols);
        @test allfinite(m_gls);
    end

    # -----------------------------
    # BayesianSDF
    # -----------------------------
    @testset "BayesianSDF" begin
        res1 = BayesianSDF(f, R, 200; psi0=5.0, d=0.5, seed=123);
        res2 = BayesianSDF(f, R, 200; psi0=5.0, d=0.5, seed=123);

        @test size(res1.lambda_path) == (200, k + 1);
        @test length(res1.R2_path) == 200;

        @test allfinite(res1.lambda_path);
        @test allfinite(res1.R2_path);

        @test isapprox(res1.lambda_path, res2.lambda_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.R2_path, res2.R2_path; rtol=1e-12, atol=1e-12);

        m = vec(mean(res1.lambda_path, dims=1));
        r2 = mean(res1.R2_path);

        @test length(m) == k + 1;
        @test allfinite(m);
        @test isfinite(r2);
    end

    # -----------------------------
    # continuous_ss_sdf
    # -----------------------------
    @testset "continuous_ss_sdf" begin
        res1 = continuous_ss_sdf(
            f, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123
        );

        res2 = continuous_ss_sdf(
            f, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123
        );

        @test size(res1.gamma_path) == (200, k);
        @test size(res1.lambda_path) == (200, k + 1);
        @test size(res1.sdf_path) == (200, t);
        @test length(res1.bma_sdf) == t;

        @test allfinite(res1.gamma_path);
        @test allfinite(res1.lambda_path);
        @test allfinite(res1.sdf_path);
        @test allfinite(res1.bma_sdf);

        @test isprobvec(vec(res1.gamma_path));
        @test all(x -> x == 0 || x == 1, vec(res1.gamma_path));

        @test isapprox(res1.gamma_path, res2.gamma_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.lambda_path, res2.lambda_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.sdf_path, res2.sdf_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.bma_sdf, res2.bma_sdf; rtol=1e-12, atol=1e-12);

        gamma_mean = vec(mean(res1.gamma_path, dims=1));
        lambda_mean = vec(mean(res1.lambda_path, dims=1));

        @test length(gamma_mean) == k;
        @test length(lambda_mean) == k + 1;
        @test isprobvec(gamma_mean);
        @test allfinite(lambda_mean);
    end

    # -----------------------------
    # continuous_ss_sdf_v2
    # -----------------------------
    @testset "continuous_ss_sdf_v2" begin
        res1 = continuous_ss_sdf_v2(
            f1, f2, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123
        );

        res2 = continuous_ss_sdf_v2(
            f1, f2, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123
        );

        @test size(res1.gamma_path) == (200, k);
        @test size(res1.lambda_path) == (200, k + 1);
        @test size(res1.sdf_path) == (200, t);
        @test length(res1.bma_sdf) == t;

        @test allfinite(res1.gamma_path);
        @test allfinite(res1.lambda_path);
        @test allfinite(res1.sdf_path);
        @test allfinite(res1.bma_sdf);

        @test isprobvec(vec(res1.gamma_path));
        @test all(x -> x == 0 || x == 1, vec(res1.gamma_path));

        @test isapprox(res1.gamma_path, res2.gamma_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.lambda_path, res2.lambda_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.sdf_path, res2.sdf_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.bma_sdf, res2.bma_sdf; rtol=1e-12, atol=1e-12);

        gamma_mean = vec(mean(res1.gamma_path, dims=1));

        @test length(gamma_mean) == k;
        @test isprobvec(gamma_mean);
    end

    # -----------------------------
    # dirac_ss_sdf_pvalue
    # -----------------------------
    @testset "dirac_ss_sdf_pvalue" begin
        res1 = dirac_ss_sdf_pvalue(f, R, 200, lambda0; seed=123);
        res2 = dirac_ss_sdf_pvalue(f, R, 200, lambda0; seed=123);

        @test size(res1.gamma_path) == (200, k);
        @test size(res1.lambda_path) == (200, k + 1);
        @test size(res1.model_probs, 2) == k + 1;

        @test allfinite(res1.gamma_path);
        @test allfinite(res1.lambda_path);
        @test allfinite(res1.model_probs);

        @test isprobvec(vec(res1.gamma_path));
        @test all(x -> x == 0 || x == 1, vec(res1.gamma_path));

        @test isapprox(res1.gamma_path, res2.gamma_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.lambda_path, res2.lambda_path; rtol=1e-12, atol=1e-12);
        @test isapprox(res1.model_probs, res2.model_probs; rtol=1e-12, atol=1e-12);

        gamma_mean = mean(res1.gamma_path);

        @test 0.0 <= gamma_mean <= 1.0;

        # Last column of model_probs should be probabilities
        probs = res1.model_probs[:, end];
        @test isprobvec(probs);
        @test isapprox(sum(probs), 1.0; atol=1e-8);
    end
end