//! beetle: on-chain offers, escrow and settlement for LoRA jobs trained in a
//! sealed sandbox.
//!
//! Sellers register, stake and post offers; buyers find an offer by account
//! filters and take it in one transaction. One job goes:
//!
//!   open_job (buyer escrows the price, the seller's bond is locked)
//!   -> start (operator declares the sandbox's one-time keys and the hash of
//!      its hardware evidence; a job with a verifier also needs its signature)
//!   -> release_key (the buyer verified the evidence locally; the DEK sealed
//!      to the sandbox key is the buyer's endorsement of those keys)
//!      or reject_evidence / claim_no_release
//!   -> submit_result (the sandbox's Ed25519 signature over the outcome is
//!      checked through the Ed25519 precompile)
//!   -> accept / reject / finalize, or timeout_start / timeout_run / abort_job
//!      (with a slash), or decline_job / abort_unstarted (no slash)
//!   -> claim (one payee at a time, so one side's account can't block the
//!      other; the last claim closes the job)
//!
//! Alongside a job, a Rendezvous account lets the two sides swap network
//! addresses for a direct link. They're sealed to each other (design notes:
//! net.md 5.7); the program stores the bytes and checks who wrote them.
//!
//! The program never judges hardware evidence: trust decisions are the
//! buyer's. It only moves money on objective facts.
//!
//! Tiers: 0 = MOCK (development only), 1 = T1 Mac App Attest, 2 = T2 TDX,
//! 3 = T3-B TPM, 4 = T3-C plain GPU, no evidence. An offer's tier is what the
//! seller claims, not a proof; a buyer who released the key on T2 checked the
//! evidence for it. So only on T2 is the sandbox's signature more than the
//! seller's word: there it releases the money, and on every other tier the
//! buyer's `accept` does. Without that, a delivered job pays at most the floor.
//!
//! Not in this version (design notes v0.2 section 11): leases, RFQ, referrers,
//! double-signing proofs, secp256r1.

use anchor_lang::compat::solana_instructions_sysvar as ix_sysvar;
use anchor_lang::prelude::*;
use anchor_spl::token_interface::{
    self, CloseAccount, Mint, TokenAccount, TokenInterface, TransferChecked,
};
use solana_sha256_hasher::hashv;

// The program id comes from the build (scripts/setup_chain.sh), so anyone can
// deploy their own from this source. Without it the id is the all-zero key,
// which no deployed program can run under.
declare_id!(Pubkey::from_str_const(match option_env!("BEETLE_PROGRAM_ID") {
    Some(id) => id,
    None => "11111111111111111111111111111111",
}));

pub const DOMAIN: &[u8] = b"beetle/v0/";
pub const SEALED_LEN: usize = 80;
/// Fixed size, so that an address blob hides how many candidates it holds.
pub const PLAIN_LEN: usize = 176;
pub const BLOB_LEN: usize = 32 + PLAIN_LEN + 16; // HPKE enc + plaintext + tag
pub const ED25519_PROGRAM_ID: Pubkey = pubkey!("Ed25519SigVerify111111111111111111111111111");
pub const INSTRUCTIONS_SYSVAR_ID: Pubkey = pubkey!("Sysvar1nstructions1111111111111111111111111");
pub const MAX_TIER: u8 = 4;
pub const TIER_T2: u8 = 2;
pub const MAX_BPS: u16 = 10_000;
pub const MAX_OFFERS: u8 = 16;
pub const MAX_OFFER_TTL: i64 = 7 * 86_400;
/// The sandbox waits for its key a fixed time from when it attests, longer than
/// this (DEK_WAIT_SECONDS in common/job_defaults.py). A longer window would let
/// a buyer release to a sandbox that has already left, at the seller's cost.
pub const MAX_RELEASE_WINDOW: i64 = 600;
/// `Seller.outage_until` of a seller that hasn't come back from a no-start.
pub const AWAY: i64 = i64::MAX;
pub const OPEN_SLOT_WINDOW: u64 = 1_500;
pub const OFFER_SCHEMA: u8 = 1;
pub const OFFER_PAUSED: u8 = 0;
pub const OFFER_ACTIVE: u8 = 1;
pub const KIND_TRAINING: u8 = 0;
pub const MAX_KIND: u8 = 2;
pub const PRICE_BY_TOKEN: u8 = 0;
pub const PRICE_BY_TIME: u8 = 1;
pub const OUTCOME_PASSED: u8 = 0;
pub const OUTCOME_FAILED: u8 = 1;
pub const OUTCOME_REQUESTER_FAULT: u8 = 2;
pub const OUTCOME_INCONCLUSIVE: u8 = 3;
pub const MAX_REACH: u8 = 0b111; // bit0 PUBLIC_UDP, bit1 ECHO, bit2 RELAY
pub const MAX_ROUNDS: u8 = 2;
pub const MAX_RDV_TTL: i64 = 86_400;
pub const RDV_MAX_POLICY: u8 = 2; // 0 direct only, 1 relay allowed, 2 relay preferred
pub const RDV_OFFERED: u8 = 0;
pub const RDV_ANSWERED: u8 = 1;
pub const RDV_FAILED: u8 = 2;

/// Lowercase hex of the chain id, the string form the job spec carries.
fn chain_tag(chain_ref: &[u8; 32]) -> [u8; 64] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 64];
    for (i, b) in chain_ref.iter().enumerate() {
        out[2 * i] = DIGITS[(b >> 4) as usize];
        out[2 * i + 1] = DIGITS[(b & 15) as usize];
    }
    out
}

/// Same construction as `digest()` in common/beetle_crypto.py, whose cluster
/// tag is the chain id in hex (see Chain.spec_chain in beetle_chain.py).
fn digest(chain_ref: &[u8; 32], type_name: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let program_id = crate::ID.to_bytes();
    let tag = chain_tag(chain_ref);
    let mut all: Vec<&[u8]> = vec![DOMAIN, type_name, &program_id, &tag];
    all.extend_from_slice(parts);
    hashv(&all).to_bytes()
}

fn bps(amount: u64, b: u64) -> u64 {
    (amount as u128 * b as u128 / MAX_BPS as u128) as u64
}

#[program]
pub mod beetle {
    use super::*;

    /// `chain_ref` is SHA256 of the CAIP-2 chain id (common/beetle_chain.py). It
    /// lives here rather than in the build so one binary serves every network.
    pub fn init_config(ctx: Context<InitConfig>, params: Params, chain_ref: [u8; 32]) -> Result<()> {
        let max = MAX_BPS;
        require!(
            params.floor_bps <= max && params.fault_bps <= max && params.min_bond_bps <= max
                && params.reject_fee_bps <= max && params.no_release_fee_bps <= max
                && params.nostart_slash_bps <= max && params.timeout_slash_bps <= max
                && params.abort_slash_bps <= max && chain_ref != [0; 32]
                && params.decline_window >= 0 && params.rdv_abort_after >= 0
                && params.settle_delay >= 0 && params.release_window > 0
                && params.release_window <= MAX_RELEASE_WINDOW
                && params.unstake_cooldown >= 0 && params.clock_margin >= 0
                && params.max_start_window > 0 && params.allowed_tiers != 0
                && params.allowed_tiers >> (MAX_TIER + 1) == 0,
            LoraError::BadParams
        );
        let c = &mut ctx.accounts.config;
        c.mint = ctx.accounts.mint.key();
        c.chain_ref = chain_ref;
        c.params = params;
        c.bump = ctx.bumps.config;
        Ok(())
    }

    // --- Sellers and stake ------------------------------------------------------

    pub fn register_seller(
        ctx: Context<RegisterSeller>,
        operator: Pubkey,
        net_pk: [u8; 32],
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let s = &mut ctx.accounts.seller;
        s.authority = ctx.accounts.authority.key();
        s.operator = operator;
        s.net_pk = net_pk;
        s.registered_at = now;
        s.stake_touched_at = now;
        s.bump = ctx.bumps.seller;
        Ok(())
    }

    /// Running jobs keep going: `start` and `submit_result` read the operator
    /// from the seller account, not from the job.
    pub fn set_operator(ctx: Context<SellerAdmin>, operator: Pubkey) -> Result<()> {
        ctx.accounts.seller.operator = operator;
        Ok(())
    }

    /// New key for sealing rendezvous offers. The old private key should be
    /// destroyed: its ciphertexts stay on the public ledger forever.
    pub fn rotate_net_pk(ctx: Context<SellerOperator>, net_pk: [u8; 32]) -> Result<()> {
        let s = &mut ctx.accounts.seller;
        s.net_pk = net_pk;
        s.net_pk_epoch += 1;
        Ok(())
    }

    pub fn stake(ctx: Context<StakeFunds>, amount: u64) -> Result<()> {
        require!(amount > 0, LoraError::BadParams);
        let now = Clock::get()?.unix_timestamp;
        let a = &ctx.accounts;
        token_interface::transfer_checked(
            CpiContext::new(
                a.token_program.key(),
                TransferChecked {
                    from: a.authority_token.to_account_info(),
                    mint: a.mint.to_account_info(),
                    to: a.vault.to_account_info(),
                    authority: a.authority.to_account_info(),
                },
            ),
            amount,
            a.mint.decimals,
        )?;
        let s = &mut ctx.accounts.seller;
        s.touch_stake(now);
        s.stake_total += amount;
        Ok(())
    }

    /// Offers that the remaining stake can't back stay posted but can't be
    /// taken: `open_job` re-checks the stake.
    pub fn request_unstake(ctx: Context<SellerAdmin>, amount: u64) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let cooldown = ctx.accounts.config.params.unstake_cooldown;
        let s = &mut ctx.accounts.seller;
        require!(amount > 0 && amount <= s.free_stake(), LoraError::InsufficientStake);
        s.unstake_pending += amount;
        s.unstake_ready_at = now + cooldown;
        Ok(())
    }

    pub fn cancel_unstake(ctx: Context<SellerAdmin>) -> Result<()> {
        let s = &mut ctx.accounts.seller;
        require!(s.unstake_pending > 0, LoraError::WrongState);
        s.unstake_pending = 0;
        s.unstake_ready_at = 0;
        Ok(())
    }

    pub fn withdraw_stake(ctx: Context<StakeFunds>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let s = &ctx.accounts.seller;
        require!(s.unstake_pending > 0, LoraError::WrongState);
        require!(now >= s.unstake_ready_at, LoraError::TooEarly);
        let amount = s.unstake_pending.min(s.stake_total.saturating_sub(s.stake_locked));
        let a = &ctx.accounts;
        pda_transfer(
            &a.token_program,
            &a.mint,
            a.vault.to_account_info(),
            a.authority_token.to_account_info(),
            a.seller.to_account_info(),
            &[b"seller", s.authority.as_ref(), &[s.bump]],
            amount,
        )?;
        let s = &mut ctx.accounts.seller;
        s.touch_stake(now);
        s.stake_total -= amount;
        s.unstake_pending = 0;
        s.unstake_ready_at = 0;
        Ok(())
    }

    // --- Offers -----------------------------------------------------------------

    pub fn post_offer(ctx: Context<PostOffer>, idx: u16, terms: OfferTerms) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let p = &ctx.accounts.config.params;
        terms.check(p, now)?;
        require!(terms.kind <= MAX_KIND, LoraError::BadParams);
        require!(
            terms.tier <= MAX_TIER && (1u8 << terms.tier) & p.allowed_tiers != 0,
            LoraError::TierNotAccepted
        );
        let (offer_stake, min_stake) = (p.offer_stake, p.min_stake[terms.tier as usize]);
        let s = &mut ctx.accounts.seller;
        require!(s.offers < MAX_OFFERS, LoraError::TooManyOffers);
        require!(s.free_stake() >= min_stake + offer_stake, LoraError::InsufficientStake);
        s.stake_locked += offer_stake;
        s.offers += 1;

        let o = &mut ctx.accounts.offer;
        o.schema = OFFER_SCHEMA;
        o.status = OFFER_ACTIVE;
        o.kind = terms.kind;
        o.tier = terms.tier;
        o.seller = ctx.accounts.seller.key();
        o.apply(&terms, now);
        o.offer_stake = offer_stake;
        o.idx = idx;
        o.bump = ctx.bumps.offer;
        Ok(())
    }

    /// New terms only reach later jobs: a job snapshots what it needs, and
    /// takers name the revision they read. Also reactivates a paused offer,
    /// and this is how a seller comes back from a no-start.
    pub fn update_offer(ctx: Context<OfferAction>, terms: OfferTerms) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        terms.check(&ctx.accounts.config.params, now)?;
        let o = &mut ctx.accounts.offer;
        require!(terms.kind == o.kind && terms.tier == o.tier, LoraError::BadParams);
        o.apply(&terms, now);
        o.status = OFFER_ACTIVE;
        o.rev += 1;
        let s = &mut ctx.accounts.seller;
        if s.outage_until == AWAY {
            s.outage_until = now;
        }
        Ok(())
    }

    /// Heartbeat: extends the offer without touching its terms or revision.
    pub fn touch_offer(ctx: Context<OfferAction>, valid_until: i64) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        require!(
            valid_until > now && valid_until <= now + MAX_OFFER_TTL,
            LoraError::BadParams
        );
        let o = &mut ctx.accounts.offer;
        o.valid_until = valid_until;
        o.updated_at = now;
        Ok(())
    }

    pub fn pause_offer(ctx: Context<OfferAction>) -> Result<()> {
        ctx.accounts.offer.status = OFFER_PAUSED;
        Ok(())
    }

    /// `settling` counts as well as the slots: a delivered job holds no slot but
    /// names this account, and closing it would leave that job unable to settle.
    pub fn close_offer(ctx: Context<CloseOffer>) -> Result<()> {
        let o = &ctx.accounts.offer;
        require!(o.slots_used == 0 && o.settling == 0, LoraError::OfferBusy);
        let offer_stake = ctx.accounts.offer.offer_stake;
        let s = &mut ctx.accounts.seller;
        s.stake_locked -= offer_stake;
        s.offers -= 1;
        Ok(())
    }

    // --- Jobs -------------------------------------------------------------------

    /// `units` is train tokens for a by-token offer and seconds for a by-time
    /// offer. The price and run budget both follow from the offer's terms, and
    /// the job keeps them: the sandbox has to sign the same numbers, so what
    /// was paid for is also the most it will run (`submit_result`).
    pub fn open_job(
        ctx: Context<OpenJob>,
        job_id: [u8; 32],
        nonce: [u8; 32],
        open_slot: u64,
        spec_hash: [u8; 32],
        units: u64,
        max_price: u64,
        expected_rev: u32,
        verifier: Pubkey,
    ) -> Result<()> {
        let clock = Clock::get()?;
        let now = clock.unix_timestamp;
        let requester = ctx.accounts.requester.key();
        // job_id = SHA256(requester || nonce || open_slot): nobody can squat
        // another requester's job id, and a closed job's id can't come back
        // once the slot has left the window. Inside the window it can; what
        // stops an old report there is sealed_dek_hash.
        require!(
            hashv(&[requester.as_ref(), &nonce, &open_slot.to_le_bytes()]).to_bytes() == job_id,
            LoraError::BadJobId
        );
        require!(
            open_slot <= clock.slot && clock.slot - open_slot <= OPEN_SLOT_WINDOW,
            LoraError::StaleOpenSlot
        );

        let p = &ctx.accounts.config.params;
        let offer = &mut ctx.accounts.offer;
        let seller = &mut ctx.accounts.seller;
        require!(offer.rev == expected_rev, LoraError::OfferChanged);
        require!(
            offer.status == OFFER_ACTIVE && now <= offer.valid_until,
            LoraError::OfferNotActive
        );
        require!(offer.kind == KIND_TRAINING, LoraError::BadParams);
        require!(offer.slots_used < offer.slots_total, LoraError::NoCapacity);
        require!(seller.outage_until != AWAY, LoraError::OfferNotActive);
        let (price, run_secs) = offer.quote(units)?;
        require!(
            price > 0 && price >= offer.min_order && price <= offer.max_order,
            LoraError::OrderOutOfRange
        );
        require!(price <= max_price, LoraError::PriceTooHigh);
        let bond = bps(price, offer.bond_bps as u64);
        require!(
            seller.free_stake() >= bond
                && seller.stake_total.saturating_sub(seller.unstake_pending)
                    >= p.min_stake[offer.tier as usize],
            LoraError::InsufficientStake
        );
        seller.stake_locked += bond;
        seller.active_jobs += 1;
        offer.slots_used += 1;

        let job = &mut ctx.accounts.job;
        job.requester = requester;
        job.host = seller.authority;
        job.payer = ctx.accounts.payer.key();
        job.offer = offer.key();
        job.verifier = verifier;
        job.job_id = job_id;
        job.spec_hash = spec_hash;
        job.price = price;
        job.bond = bond;
        job.state = JobState::Opened;
        job.opened_at = now;
        job.start_by = now + offer.start_window_s as i64;
        job.run_secs = run_secs;
        job.units = units;
        job.grace_s = offer.grace_s;
        job.price_mode = offer.price_mode;
        job.measurement = offer.measurement;
        job.offer_rev = offer.rev;
        job.floor_bps = p.floor_bps;
        job.fault_bps = p.fault_bps;
        job.tier = offer.tier;
        job.rdv_opened = false;
        job.bump = ctx.bumps.job;

        let a = &ctx.accounts;
        token_interface::transfer_checked(
            CpiContext::new(
                a.token_program.key(),
                TransferChecked {
                    from: a.requester_token.to_account_info(),
                    mint: a.mint.to_account_info(),
                    to: a.vault.to_account_info(),
                    authority: a.requester.to_account_info(),
                },
            ),
            price,
            a.mint.decimals,
        )
    }

    /// The operator declares the sandbox keys and commits to its evidence. The
    /// program can't read the evidence; the buyer verifies it off chain and
    /// endorses these keys by releasing the DEK. A job with a verifier also
    /// requires that verifier's signature over the registration, in the
    /// previous instruction.
    pub fn start(
        ctx: Context<StartJob>,
        sign_pk: [u8; 32],
        enc_pk: [u8; 32],
        measurement: [u8; 32],
        evidence_hash: [u8; 32],
        expiry: i64,
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let release_window = ctx.accounts.config.params.release_window;
        let chain_ref = ctx.accounts.config.chain_ref;
        let job = &mut ctx.accounts.job;
        require!(job.state == JobState::Opened, LoraError::WrongState);
        require!(now <= job.start_by, LoraError::TooLate);
        require!(measurement == job.measurement, LoraError::MeasurementMismatch);
        if job.verifier != Pubkey::default() {
            let msg = digest(
                &chain_ref,
                b"register",
                &[
                    &job.job_id,
                    &job.spec_hash,
                    &sign_pk,
                    &enc_pk,
                    &measurement,
                    &[job.tier],
                    &expiry.to_le_bytes(),
                    job.host.as_ref(),
                ],
            );
            verify_prev_ed25519(
                &ctx.accounts.instructions.to_account_info(),
                &job.verifier.to_bytes(),
                &msg,
            )?;
            require!(now <= expiry, LoraError::RegistrationExpired);
        }
        job.sign_pk = sign_pk;
        job.enc_pk = enc_pk;
        job.evidence_hash = evidence_hash;
        job.release_by = now + release_window;
        job.state = JobState::Started;
        Ok(())
    }

    /// `evidence_hash` is what the buyer verified locally; it must be what the
    /// operator committed to. The run budget starts here, so a late release
    /// doesn't eat into the seller's time.
    pub fn release_key(
        ctx: Context<RequesterAction>,
        sealed_dek: [u8; SEALED_LEN],
        evidence_hash: [u8; 32],
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let job = &mut ctx.accounts.job;
        require!(job.state == JobState::Started, LoraError::WrongState);
        require!(now <= job.release_by, LoraError::TooLate);
        require!(evidence_hash == job.evidence_hash, LoraError::EvidenceMismatch);
        job.sealed_dek = sealed_dek;
        job.has_dek = true;
        job.deadline = now + job.run_secs;
        job.state = JobState::Running;
        Ok(())
    }

    /// The buyer's evidence check failed: refund less a small capped fee.
    pub fn reject_evidence(ctx: Context<EndJob>) -> Result<()> {
        require_keys_eq!(ctx.accounts.caller.key(), ctx.accounts.job.requester, LoraError::NotRequester);
        require!(ctx.accounts.job.state == JobState::Started, LoraError::WrongState);
        require!(
            Clock::get()?.unix_timestamp <= ctx.accounts.job.release_by,
            LoraError::TooLate
        );
        let p = &ctx.accounts.config.params;
        let fee = bps(ctx.accounts.job.price, p.reject_fee_bps as u64).min(p.reject_fee_cap);
        ctx.accounts.seller.n_rejected += 1;
        let refund = ctx.accounts.job.price - fee;
        end_job(ctx.accounts, JobState::Rejected, fee, refund, 0);
        Ok(())
    }

    /// The buyer let the release window pass: the seller is paid for holding
    /// the capacity, and the bond and slot are freed.
    pub fn claim_no_release(ctx: Context<EndJob>) -> Result<()> {
        let p = &ctx.accounts.config.params;
        let job = &ctx.accounts.job;
        require!(job.state == JobState::Started, LoraError::WrongState);
        require!(
            Clock::get()?.unix_timestamp > job.release_by + p.clock_margin,
            LoraError::TooEarly
        );
        let fee = bps(job.price, p.no_release_fee_bps as u64).min(p.no_release_fee_cap);
        let refund = job.price - fee;
        ctx.accounts.seller.n_norelease += 1;
        end_job(ctx.accounts, JobState::NoRelease, fee, refund, 0);
        Ok(())
    }

    /// Only the outcome and hashes are public; scores stay in the report the
    /// sandbox sealed to the requester. `sealed_dek_hash` is the hash of the
    /// sealed key the sandbox actually received, so a host that feeds it a
    /// fake key can't get a "bad key" report through, and a report can't be
    /// replayed into a reopened job (each release seals afresh). The signature
    /// also covers the terms read from the job, so a sandbox told it was paid
    /// for more than this order can't settle it. The adapter key only ever
    /// reaches the buyer here: past the deadline no key, so no refund on top.
    ///
    /// The sandbox is done, so the job's slot can take another order while the
    /// buyer takes its time. The bond stays locked until the job settles, so
    /// what backs orders is still the stake: more slots don't mean more risk.
    /// The offer is an optional extra account, not part of `SubmitResult`, so
    /// that a client without it still submits; its slot then waits for settling.
    pub fn submit_result(
        ctx: Context<SubmitResult>,
        outcome: u8,
        work_bps: u16,
        sealed_dek_hash: [u8; 32],
        adapter_ct_hash: [u8; 32],
        has_adapter_key: bool,
        sealed_adapter_key: [u8; SEALED_LEN],
        report_hash: [u8; 32],
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let settle_delay = ctx.accounts.config.params.settle_delay;
        let chain_ref = ctx.accounts.config.chain_ref;
        let job = &mut ctx.accounts.job;
        require!(job.state == JobState::Running, LoraError::WrongState);
        require!(now <= job.deadline, LoraError::TooLate);
        require!(
            outcome <= OUTCOME_INCONCLUSIVE
                && work_bps <= MAX_BPS
                && (outcome == OUTCOME_PASSED) == has_adapter_key,
            LoraError::BadParams
        );
        require!(
            sealed_dek_hash == hashv(&[&job.sealed_dek]).to_bytes(),
            LoraError::SealedDekMismatch
        );

        let msg = digest(
            &chain_ref,
            b"settle",
            &[
                &job.job_id,
                &[outcome],
                &work_bps.to_le_bytes(),
                &sealed_dek_hash,
                &adapter_ct_hash,
                &[has_adapter_key as u8],
                &sealed_adapter_key,
                &report_hash,
                &job.units.to_le_bytes(),
                &[job.price_mode],
                &job.run_secs.to_le_bytes(),
                &job.grace_s.to_le_bytes(),
            ],
        );
        verify_prev_ed25519(&ctx.accounts.instructions.to_account_info(), &job.sign_pk, &msg)?;

        job.outcome = outcome;
        job.work_bps = work_bps;
        job.adapter_ct_hash = adapter_ct_hash;
        job.has_adapter_key = has_adapter_key;
        job.sealed_adapter_key = sealed_adapter_key;
        job.report_hash = report_hash;
        job.challenge_end = now + settle_delay;
        job.state = JobState::Delivered;
        match ctx.remaining_accounts.first() {
            Some(offer) => release_slot(offer, ctx.accounts.job.offer),
            None => Ok(()),
        }
    }

    /// The buyer checked the delivery and takes it. This is also what releases
    /// the full price where no proven sandbox vouches for the result.
    pub fn accept(ctx: Context<EndJob>) -> Result<()> {
        require_keys_eq!(ctx.accounts.caller.key(), ctx.accounts.job.requester, LoraError::NotRequester);
        require!(ctx.accounts.job.state == JobState::Delivered, LoraError::WrongState);
        settle(ctx.accounts, true);
        Ok(())
    }

    /// The buyer checked a delivery no proven sandbox vouches for, and it failed
    /// on its hidden questions. Pays what `finalize` would, now.
    pub fn reject(ctx: Context<EndJob>) -> Result<()> {
        require_keys_eq!(ctx.accounts.caller.key(), ctx.accounts.job.requester, LoraError::NotRequester);
        require!(ctx.accounts.job.state == JobState::Delivered, LoraError::WrongState);
        require!(ctx.accounts.job.tier != TIER_T2, LoraError::ProvenResult);
        settle(ctx.accounts, false);
        Ok(())
    }

    /// Anyone can settle once the settle delay has passed. Silence from the
    /// buyer isn't an acceptance: see `payout`.
    pub fn finalize(ctx: Context<EndJob>) -> Result<()> {
        let job = &ctx.accounts.job;
        require!(job.state == JobState::Delivered, LoraError::WrongState);
        require!(Clock::get()?.unix_timestamp >= job.challenge_end, LoraError::TooEarly);
        settle(ctx.accounts, false);
        Ok(())
    }

    /// The seller turns down an order it can't take, any time before the
    /// no-start slash could land: a buyer who stalls, names a verifier that
    /// won't sign or orders more than it paid for can't collect that slash from
    /// a seller who is there. Full refund, no slash, on the record as declined.
    pub fn decline_job(ctx: Context<EndJob>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let a = &ctx.accounts;
        require_keys_eq!(a.caller.key(), a.seller.operator, LoraError::NotOperator);
        require!(a.job.state == JobState::Opened, LoraError::WrongState);
        require!(now <= a.job.start_by + a.config.params.clock_margin, LoraError::TooLate);
        let refund = a.job.price;
        ctx.accounts.seller.n_declined += 1;
        end_job(ctx.accounts, JobState::Declined, 0, refund, 0);
        Ok(())
    }

    /// The seller gives up on a running job: full refund, and a smaller slash
    /// than waiting for the deadline would cost.
    pub fn abort_job(ctx: Context<EndJob>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let a = &ctx.accounts;
        require_keys_eq!(a.caller.key(), a.seller.operator, LoraError::NotOperator);
        require!(a.job.state == JobState::Running, LoraError::WrongState);
        let slash = bps(a.job.bond, a.config.params.abort_slash_bps as u64);
        let refund = a.job.price;
        let s = &mut ctx.accounts.seller;
        s.n_abort += 1;
        s.slash(slash, now);
        end_job(ctx.accounts, JobState::Aborted, 0, refund, slash);
        Ok(())
    }

    /// Either side, once a rendezvous was opened and the job still hasn't
    /// started: no chain can tell whose network failed or whether the inputs
    /// were sent, so there's no slash. A seller that never answered gets the
    /// failed connection on its record. One that did answer was there, and a
    /// buyer who leaves after reading its address pays a small flat fee, so
    /// collecting addresses isn't free. The time and the flags are on the job,
    /// not the rendezvous, because the buyer controls that account: reopening
    /// or closing it must not be able to hold a present seller in place.
    pub fn abort_unstarted(ctx: Context<EndJob>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let a = &ctx.accounts;
        let caller = a.caller.key();
        require!(
            caller == a.job.requester || caller == a.seller.operator,
            LoraError::NotParticipant
        );
        require!(a.job.state == JobState::Opened, LoraError::WrongState);
        require!(a.job.rdv_opened, LoraError::NoRendezvous);
        require!(
            now >= a.job.opened_at + a.config.params.rdv_abort_after,
            LoraError::TooEarly
        );
        let answered = a.job.rdv_answered;
        let fee = if answered && caller == a.job.requester {
            a.config.params.reject_fee_cap.min(a.job.price)
        } else {
            0
        };
        let refund = a.job.price - fee;
        if !answered {
            ctx.accounts.seller.n_connect_fail += 1;
        }
        end_job(ctx.accounts, JobState::Unstarted, fee, refund, 0);
        Ok(())
    }

    /// The seller never started: full refund and a slash. One outage costs the
    /// seller once, however many offers and slots it had: all its offers stop
    /// taking orders until it comes back, and a job due before that is the same
    /// outage, whenever someone gets round to calling this.
    pub fn timeout_start(ctx: Context<EndJob>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let p = &ctx.accounts.config.params;
        let job = &ctx.accounts.job;
        require!(job.state == JobState::Opened, LoraError::WrongState);
        require!(now > job.start_by + p.clock_margin, LoraError::TooEarly);
        let slash = if job.start_by > ctx.accounts.seller.outage_until {
            bps(job.bond, p.nostart_slash_bps as u64).max(p.nostart_slash_min).min(job.bond)
        } else {
            0
        };
        let refund = job.price;
        let s = &mut ctx.accounts.seller;
        s.n_nostart += 1;
        s.slash(slash, now);
        s.outage_until = AWAY;
        ctx.accounts.offer.status = OFFER_PAUSED;
        end_job(ctx.accounts, JobState::NoStart, 0, refund, slash);
        Ok(())
    }

    /// The seller got the key but delivered nothing in time: full refund and a
    /// bigger slash.
    pub fn timeout_run(ctx: Context<EndJob>) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let p = &ctx.accounts.config.params;
        let job = &ctx.accounts.job;
        require!(job.state == JobState::Running, LoraError::WrongState);
        require!(now > job.deadline + p.clock_margin, LoraError::TooEarly);
        let slash = bps(job.bond, p.timeout_slash_bps as u64);
        let refund = job.price;
        let s = &mut ctx.accounts.seller;
        s.n_timeout += 1;
        s.slash(slash, now);
        end_job(ctx.accounts, JobState::TimedOut, 0, refund, slash);
        Ok(())
    }

    /// Pays one payee what a finished job owes them. Anyone can call it, and
    /// each payee is a separate call: one frozen or closed account blocks only
    /// its own claim. The last claim closes the vault and the job, returning
    /// their rent to whoever paid it.
    pub fn claim(ctx: Context<Claim>, who: Payee) -> Result<()> {
        let a = &ctx.accounts;
        let job = &a.job;
        require!(job.state.is_finished(), LoraError::WrongState);
        let payee = if who == Payee::Seller { job.host } else { job.requester };
        require_keys_eq!(a.to.owner, payee, LoraError::WrongPayee);
        let amount = match who {
            Payee::Seller => job.seller_due,
            Payee::Buyer => job.buyer_due,
            Payee::Slash => job.slash_due,
        };
        require!(amount > 0, LoraError::NothingDue);
        let job_id = job.job_id;
        let job_bump = [job.bump];
        let job_seeds: &[&[u8]] = &[b"job", job_id.as_ref(), &job_bump];
        if who == Payee::Slash {
            let seller_bump = [a.seller.bump];
            pda_transfer(
                &a.token_program,
                &a.mint,
                a.seller_vault.to_account_info(),
                a.to.to_account_info(),
                a.seller.to_account_info(),
                &[b"seller", job.host.as_ref(), &seller_bump],
                amount,
            )?;
        } else {
            pda_transfer(
                &a.token_program,
                &a.mint,
                a.vault.to_account_info(),
                a.to.to_account_info(),
                a.job.to_account_info(),
                job_seeds,
                amount,
            )?;
        }

        let job = &mut ctx.accounts.job;
        match who {
            Payee::Seller => job.seller_due = 0,
            Payee::Buyer => job.buyer_due = 0,
            Payee::Slash => job.slash_due = 0,
        }
        if job.seller_due > 0 || job.buyer_due > 0 || job.slash_due > 0 {
            return Ok(());
        }

        // Anything left is tokens someone sent to the vault; it goes to the
        // last payee so that nobody can keep the vault from closing.
        ctx.accounts.vault.reload()?;
        let a = &ctx.accounts;
        if a.vault.amount > 0 {
            pda_transfer(
                &a.token_program,
                &a.mint,
                a.vault.to_account_info(),
                a.to.to_account_info(),
                a.job.to_account_info(),
                job_seeds,
                a.vault.amount,
            )?;
        }
        token_interface::close_account(
            CpiContext::new(
                a.token_program.key(),
                CloseAccount {
                    account: a.vault.to_account_info(),
                    destination: a.rent_receiver.to_account_info(),
                    authority: a.job.to_account_info(),
                },
            )
            .with_signer(&[job_seeds]),
        )?;
        ctx.accounts.job.close(ctx.accounts.rent_receiver.to_account_info())
    }

    // --- Rendezvous -------------------------------------------------------------

    /// The buyer starts a connection and writes the first offer. A later call
    /// reuses the account and starts over at round 1: MAX_ROUNDS limits one
    /// connection, and a job can need several (training runs for hours).
    pub fn rdv_open(
        ctx: Context<RdvOpen>,
        policy: u8,
        ttl_secs: i64,
        offer: [u8; BLOB_LEN],
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        require!(
            policy <= RDV_MAX_POLICY && ttl_secs > 0 && ttl_secs <= MAX_RDV_TTL,
            LoraError::BadParams
        );
        let job = &mut ctx.accounts.job;
        require!(
            matches!(
                job.state,
                JobState::Opened | JobState::Started | JobState::Running | JobState::Delivered
            ),
            LoraError::WrongState
        );
        job.rdv_opened = true;
        let (job_id, host) = (job.job_id, job.host);
        let rdv = &mut ctx.accounts.rdv;
        if rdv.round == 0 {
            rdv.job_id = job_id;
            rdv.requester = ctx.accounts.requester.key();
            rdv.host = host;
            rdv.payer = ctx.accounts.payer.key();
            rdv.bump = ctx.bumps.rdv;
        }
        rdv.opened_at = now;
        rdv.expires_at = now + ttl_secs;
        rdv.policy = policy;
        rdv.round = 1;
        rdv.state = RDV_OFFERED;
        rdv.offer = offer;
        rdv.answer = [0; BLOB_LEN];
        Ok(())
    }

    pub fn rdv_answer(ctx: Context<RdvAnswer>, round: u8, answer: [u8; BLOB_LEN]) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let rdv = &mut ctx.accounts.rdv;
        require!(rdv.state == RDV_OFFERED, LoraError::WrongState);
        require!(round == rdv.round, LoraError::BadRound);
        require!(now <= rdv.expires_at, LoraError::TooLate);
        rdv.answer = answer;
        rdv.state = RDV_ANSWERED;
        ctx.accounts.job.rdv_answered = true;
        Ok(())
    }

    /// A new round with fresh addresses, for when the dial failed or the seller
    /// never answered. It gets a full time-to-live of its own.
    pub fn rdv_reoffer(ctx: Context<RdvReoffer>, round: u8, offer: [u8; BLOB_LEN]) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let rdv = &mut ctx.accounts.rdv;
        require!(rdv.state != RDV_FAILED, LoraError::WrongState);
        require!(round == rdv.round + 1 && round <= MAX_ROUNDS, LoraError::BadRound);
        let ttl = rdv.expires_at - rdv.opened_at;
        rdv.opened_at = now;
        rdv.expires_at = now + ttl;
        rdv.round = round;
        rdv.state = RDV_OFFERED;
        rdv.offer = offer;
        rdv.answer = [0; BLOB_LEN];
        Ok(())
    }

    /// Either side gives up on this connection. Nothing here moves money.
    pub fn rdv_fail(ctx: Context<RdvFail>, reason: u8) -> Result<()> {
        let caller = ctx.accounts.caller.key();
        let rdv = &mut ctx.accounts.rdv;
        require!(
            caller == rdv.requester || caller == ctx.accounts.seller.operator,
            LoraError::NotParticipant
        );
        require!(rdv.state != RDV_FAILED, LoraError::WrongState);
        rdv.state = RDV_FAILED;
        msg!("rendezvous failed, reason {}", reason);
        Ok(())
    }

    /// Returns the deposit to whoever paid it. The buyer can close any time;
    /// once it failed or expired, anyone can.
    pub fn rdv_close(ctx: Context<RdvClose>) -> Result<()> {
        let rdv = &ctx.accounts.rdv;
        require!(
            ctx.accounts.caller.key() == rdv.requester
                || rdv.state == RDV_FAILED
                || Clock::get()?.unix_timestamp > rdv.expires_at,
            LoraError::TooEarly
        );
        Ok(())
    }
}

/// What a delivered job owes (seller, buyer). The signed outcome decides, and
/// a failed job pays the compute floor in proportion to the work actually
/// done, so a host can't throttle the sandbox and collect it. The full price
/// takes more than a signature: off T2 the seller holds the signing key, so
/// `Passed` is only its claim until the buyer accepts it. A buyer's bad inputs
/// pay the fault fee or, on T2, the work they used up if that's more.
fn payout(job: &Job, accepted: bool) -> (u64, u64) {
    let proven = job.tier == TIER_T2;
    let floor = job.price as u128 * job.floor_bps as u128 * job.work_bps as u128;
    let floor = (floor / (MAX_BPS as u128 * MAX_BPS as u128)) as u64;
    let fault = bps(job.price, job.fault_bps as u64);
    let seller = match job.outcome {
        OUTCOME_PASSED if proven || accepted => job.price,
        OUTCOME_PASSED | OUTCOME_FAILED => floor,
        OUTCOME_REQUESTER_FAULT if proven => fault.max(bps(job.price, job.work_bps as u64)),
        OUTCOME_REQUESTER_FAULT => fault,
        _ => 0,
    };
    (seller, job.price - seller)
}

fn settle(a: &mut EndJob, accepted: bool) {
    let (seller_due, buyer_due) = payout(&a.job, accepted);
    let s = &mut a.seller;
    s.volume += seller_due;
    match a.job.outcome {
        OUTCOME_PASSED if seller_due == a.job.price => s.n_settled += 1,
        OUTCOME_PASSED | OUTCOME_FAILED => s.n_failed_accept += 1,
        _ => {}
    }
    end_job(a, JobState::Settled, seller_due, buyer_due, 0);
}

/// A delivered job's slot goes back to the offer, which remembers that the job
/// is still to settle. Out of line, to keep the offer off submit_result's stack.
#[inline(never)]
fn release_slot<'info>(info: &'info AccountInfo<'info>, job_offer: Pubkey) -> Result<()> {
    require_keys_eq!(info.key(), job_offer, LoraError::BadParams);
    let mut offer = Box::new(Account::<Offer>::try_from(info)?);
    offer.slots_used -= 1;
    offer.settling += 1;
    offer.exit(&crate::ID)
}

/// Book the end of a job: free the bond and the slot and record what is owed.
/// Nothing moves here, so no payee's account can block this step. A delivered
/// job gave its slot back already, unless a client didn't pass the offer or it
/// was delivered before release_slot existed. It doesn't matter which job is
/// which: `slots_used + settling` counts the offer's unfinished jobs, and a
/// delivered job is only counted in `slots_used` once `settling` has run out.
fn end_job(a: &mut EndJob, state: JobState, seller_due: u64, buyer_due: u64, slash_due: u64) {
    let s = &mut a.seller;
    s.stake_locked -= a.job.bond;
    s.active_jobs -= 1;
    if a.job.state == JobState::Delivered && a.offer.settling > 0 {
        a.offer.settling -= 1;
    } else {
        a.offer.slots_used -= 1;
    }
    let job = &mut a.job;
    job.seller_due = seller_due;
    job.buyer_due = buyer_due;
    job.slash_due = slash_due;
    job.state = state;
}

/// Token transfer out of an account whose authority is a PDA of this program.
fn pda_transfer<'info>(
    token_program: &Interface<'info, TokenInterface>,
    mint: &InterfaceAccount<'info, Mint>,
    from: AccountInfo<'info>,
    to: AccountInfo<'info>,
    authority: AccountInfo<'info>,
    seeds: &[&[u8]],
    amount: u64,
) -> Result<()> {
    token_interface::transfer_checked(
        CpiContext::new(token_program.key(), TransferChecked { from, mint: mint.to_account_info(), to, authority })
            .with_signer(&[seeds]),
        amount,
        mint.decimals,
    )
}

/// Check that the previous instruction is an Ed25519 precompile call that
/// verified `msg` under `pk`, with all offsets pointing into itself.
/// Skipping the offset checks lets an attacker point the precompile at
/// unrelated data, a known class of Solana bugs.
fn verify_prev_ed25519(ix_acc: &AccountInfo, pk: &[u8; 32], msg: &[u8; 32]) -> Result<()> {
    let current = ix_sysvar::load_current_index_checked(ix_acc)? as usize;
    require!(current > 0, LoraError::MissingSigIx);
    let ix = ix_sysvar::load_instruction_at_checked(current - 1, ix_acc)?;
    require_keys_eq!(ix.program_id, ED25519_PROGRAM_ID, LoraError::MissingSigIx);

    let d = &ix.data;
    require!(d.len() >= 16 && d[0] == 1, LoraError::BadSigIx);
    let u = |i: usize| u16::from_le_bytes([d[i], d[i + 1]]);
    let (sig_off, sig_ix, pk_off, pk_ix, msg_off, msg_len, msg_ix) =
        (u(2) as usize, u(4), u(6) as usize, u(8), u(10) as usize, u(12) as usize, u(14));
    require!(
        sig_ix == u16::MAX && pk_ix == u16::MAX && msg_ix == u16::MAX,
        LoraError::BadSigIx
    );
    require!(
        msg_len == 32 && d.len() >= sig_off + 64 && d.len() >= pk_off + 32 && d.len() >= msg_off + 32,
        LoraError::BadSigIx
    );
    require!(&d[pk_off..pk_off + 32] == pk, LoraError::WrongSigner);
    require!(&d[msg_off..msg_off + 32] == msg, LoraError::WrongMessage);
    Ok(())
}

// --- Accounts ----------------------------------------------------------------

/// Only the upgrade authority can initialize, so a deployment can't be
/// front-run with someone else's mint and parameters.
#[derive(Accounts)]
pub struct InitConfig<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(init, payer = authority, space = 8 + Config::INIT_SPACE, seeds = [b"config"], bump)]
    pub config: Box<Account<'info, Config>>,
    pub mint: Box<InterfaceAccount<'info, Mint>>,
    #[account(constraint = program.programdata_address()? == Some(program_data.key()) @ LoraError::NotUpgradeAuthority)]
    pub program: Program<'info, crate::program::Beetle>,
    #[account(constraint = program_data.upgrade_authority_address == Some(authority.key()) @ LoraError::NotUpgradeAuthority)]
    pub program_data: Box<Account<'info, ProgramData>>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterSeller<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    pub authority: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump, has_one = mint)]
    pub config: Box<Account<'info, Config>>,
    #[account(
        init, payer = payer, space = 8 + Seller::INIT_SPACE,
        seeds = [b"seller", authority.key().as_ref()], bump
    )]
    pub seller: Box<Account<'info, Seller>>,
    #[account(
        init, payer = payer, seeds = [b"seller_vault", seller.key().as_ref()], bump,
        token::mint = mint, token::authority = seller, token::token_program = token_program
    )]
    pub vault: Box<InterfaceAccount<'info, TokenAccount>>,
    pub mint: Box<InterfaceAccount<'info, Mint>>,
    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SellerAdmin<'info> {
    pub authority: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"seller", authority.key().as_ref()], bump = seller.bump, has_one = authority)]
    pub seller: Box<Account<'info, Seller>>,
}

#[derive(Accounts)]
pub struct SellerOperator<'info> {
    pub operator: Signer<'info>,
    #[account(mut, seeds = [b"seller", seller.authority.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
}

#[derive(Accounts)]
pub struct StakeFunds<'info> {
    pub authority: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump, has_one = mint)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"seller", authority.key().as_ref()], bump = seller.bump, has_one = authority)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(
        mut, seeds = [b"seller_vault", seller.key().as_ref()], bump,
        token::mint = mint, token::authority = seller
    )]
    pub vault: Box<InterfaceAccount<'info, TokenAccount>>,
    pub mint: Box<InterfaceAccount<'info, Mint>>,
    #[account(mut, token::mint = mint, token::authority = authority)]
    pub authority_token: Box<InterfaceAccount<'info, TokenAccount>>,
    pub token_program: Interface<'info, TokenInterface>,
}

#[derive(Accounts)]
#[instruction(idx: u16)]
pub struct PostOffer<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    pub operator: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"seller", seller.authority.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(
        init, payer = payer, space = 8 + Offer::INIT_SPACE,
        seeds = [b"offer", seller.key().as_ref(), &idx.to_le_bytes()], bump
    )]
    pub offer: Box<Account<'info, Offer>>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct OfferAction<'info> {
    pub operator: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"seller", seller.authority.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, has_one = seller)]
    pub offer: Box<Account<'info, Offer>>,
}

#[derive(Accounts)]
pub struct CloseOffer<'info> {
    pub operator: Signer<'info>,
    #[account(mut, seeds = [b"seller", seller.authority.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, has_one = seller, close = rent_receiver)]
    pub offer: Box<Account<'info, Offer>>,
    /// CHECK: the seller's own wallet gets the rent back.
    #[account(mut, address = seller.authority)]
    pub rent_receiver: UncheckedAccount<'info>,
}

#[derive(Accounts)]
#[instruction(job_id: [u8; 32])]
pub struct OpenJob<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    pub requester: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump, has_one = mint)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, has_one = seller)]
    pub offer: Box<Account<'info, Offer>>,
    #[account(mut, seeds = [b"seller", seller.authority.as_ref()], bump = seller.bump)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(init, payer = payer, space = 8 + Job::INIT_SPACE, seeds = [b"job", job_id.as_ref()], bump)]
    pub job: Box<Account<'info, Job>>,
    #[account(
        init, payer = payer, seeds = [b"vault", job_id.as_ref()], bump,
        token::mint = mint, token::authority = job, token::token_program = token_program
    )]
    pub vault: Box<InterfaceAccount<'info, TokenAccount>>,
    pub mint: Box<InterfaceAccount<'info, Mint>>,
    #[account(mut, token::mint = mint, token::authority = requester)]
    pub requester_token: Box<InterfaceAccount<'info, TokenAccount>>,
    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct StartJob<'info> {
    pub operator: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(seeds = [b"seller", job.host.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump)]
    pub job: Box<Account<'info, Job>>,
    /// CHECK: address-constrained to the instructions sysvar.
    #[account(address = INSTRUCTIONS_SYSVAR_ID)]
    pub instructions: UncheckedAccount<'info>,
}

#[derive(Accounts)]
pub struct RequesterAction<'info> {
    pub requester: Signer<'info>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump, has_one = requester)]
    pub job: Box<Account<'info, Job>>,
}

#[derive(Accounts)]
pub struct SubmitResult<'info> {
    pub operator: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(seeds = [b"seller", job.host.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump)]
    pub job: Box<Account<'info, Job>>,
    /// CHECK: address-constrained to the instructions sysvar.
    #[account(address = INSTRUCTIONS_SYSVAR_ID)]
    pub instructions: UncheckedAccount<'info>,
}

/// Every way a job can end. No token account appears here: these steps only
/// write down what is owed, and `claim` pays it.
#[derive(Accounts)]
pub struct EndJob<'info> {
    pub caller: Signer<'info>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump, has_one = offer)]
    pub job: Box<Account<'info, Job>>,
    #[account(mut, seeds = [b"seller", job.host.as_ref()], bump = seller.bump)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut)]
    pub offer: Box<Account<'info, Offer>>,
}

#[derive(Accounts)]
pub struct Claim<'info> {
    #[account(seeds = [b"config"], bump = config.bump, has_one = mint)]
    pub config: Box<Account<'info, Config>>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump)]
    pub job: Box<Account<'info, Job>>,
    #[account(mut, seeds = [b"vault", job.job_id.as_ref()], bump, token::mint = mint, token::authority = job)]
    pub vault: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(seeds = [b"seller", job.host.as_ref()], bump = seller.bump)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(
        mut, seeds = [b"seller_vault", seller.key().as_ref()], bump,
        token::mint = mint, token::authority = seller
    )]
    pub seller_vault: Box<InterfaceAccount<'info, TokenAccount>>,
    pub mint: Box<InterfaceAccount<'info, Mint>>,
    /// The handler checks that this belongs to the payee named by `who`.
    #[account(mut, token::mint = mint)]
    pub to: Box<InterfaceAccount<'info, TokenAccount>>,
    /// CHECK: receives the vault's and the job's rent; must be whoever paid it.
    #[account(mut, address = job.payer)]
    pub rent_receiver: UncheckedAccount<'info>,
    pub token_program: Interface<'info, TokenInterface>,
}

#[derive(Accounts)]
pub struct RdvOpen<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    pub requester: Signer<'info>,
    #[account(mut, seeds = [b"job", job.job_id.as_ref()], bump = job.bump, has_one = requester)]
    pub job: Box<Account<'info, Job>>,
    #[account(
        init_if_needed, payer = payer, space = 8 + Rendezvous::INIT_SPACE,
        seeds = [b"rdv", job.job_id.as_ref()], bump
    )]
    pub rdv: Box<Account<'info, Rendezvous>>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RdvAnswer<'info> {
    pub operator: Signer<'info>,
    #[account(seeds = [b"seller", rdv.host.as_ref()], bump = seller.bump, has_one = operator)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, seeds = [b"rdv", rdv.job_id.as_ref()], bump = rdv.bump)]
    pub rdv: Box<Account<'info, Rendezvous>>,
    #[account(mut, seeds = [b"job", rdv.job_id.as_ref()], bump = job.bump)]
    pub job: Box<Account<'info, Job>>,
}

#[derive(Accounts)]
pub struct RdvReoffer<'info> {
    pub requester: Signer<'info>,
    #[account(mut, seeds = [b"rdv", rdv.job_id.as_ref()], bump = rdv.bump, has_one = requester)]
    pub rdv: Box<Account<'info, Rendezvous>>,
}

#[derive(Accounts)]
pub struct RdvFail<'info> {
    pub caller: Signer<'info>,
    #[account(seeds = [b"seller", rdv.host.as_ref()], bump = seller.bump)]
    pub seller: Box<Account<'info, Seller>>,
    #[account(mut, seeds = [b"rdv", rdv.job_id.as_ref()], bump = rdv.bump)]
    pub rdv: Box<Account<'info, Rendezvous>>,
}

#[derive(Accounts)]
pub struct RdvClose<'info> {
    pub caller: Signer<'info>,
    #[account(mut, seeds = [b"rdv", rdv.job_id.as_ref()], bump = rdv.bump, close = rent_receiver)]
    pub rdv: Box<Account<'info, Rendezvous>>,
    /// CHECK: receives the deposit; must be whoever paid it.
    #[account(mut, address = rdv.payer)]
    pub rent_receiver: UncheckedAccount<'info>,
}

// --- State -------------------------------------------------------------------

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, InitSpace)]
pub struct Params {
    pub floor_bps: u16,
    pub fault_bps: u16,
    pub min_bond_bps: u16,
    pub reject_fee_bps: u16,
    pub no_release_fee_bps: u16,
    pub nostart_slash_bps: u16,
    pub timeout_slash_bps: u16,
    pub reject_fee_cap: u64,
    pub no_release_fee_cap: u64,
    pub nostart_slash_min: u64,
    pub offer_stake: u64,
    pub min_stake: [u64; 5],
    pub settle_delay: i64,
    pub release_window: i64,
    pub unstake_cooldown: i64,
    /// Extra seconds before a timeout can fire, for chain clock drift.
    pub clock_margin: i64,
    pub max_start_window: u32,
    pub allowed_tiers: u8,
    /// How long after open_job a seller waits for a buyer it hasn't heard
    /// from before declining. The program doesn't use this: it's the published
    /// number both clients go by. A decline itself works until `start_by`.
    pub decline_window: i64,
    /// How long after open_job either side can call off an unstarted job.
    pub rdv_abort_after: i64,
    pub abort_slash_bps: u16,
}

#[account]
#[derive(InitSpace)]
pub struct Config {
    pub mint: Pubkey,
    pub chain_ref: [u8; 32],
    pub params: Params,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct Seller {
    pub authority: Pubkey,
    pub operator: Pubkey,
    /// X25519 key that rendezvous offers are sealed to.
    pub net_pk: [u8; 32],
    pub net_pk_epoch: u32,
    pub stake_total: u64,
    pub stake_locked: u64,
    pub unstake_pending: u64,
    pub unstake_ready_at: i64,
    pub registered_at: i64,
    pub stake_usd_days: u64,
    pub stake_touched_at: i64,
    pub offers: u8,
    pub active_jobs: u16,
    pub n_settled: u32,
    pub n_failed_accept: u32,
    pub n_nostart: u32,
    pub n_timeout: u32,
    pub n_rejected: u32,
    pub n_norelease: u32,
    pub n_declined: u32,
    pub n_abort: u32,
    pub n_connect_fail: u32,
    pub volume: u64,
    pub last_fail_at: i64,
    /// AWAY since a no-start, then when the operator came back. A job due
    /// before this belongs to an outage that was already slashed.
    pub outage_until: i64,
    pub bump: u8,
    pub reserved: [u8; 12],
}

impl Seller {
    fn free_stake(&self) -> u64 {
        self.stake_total.saturating_sub(self.stake_locked).saturating_sub(self.unstake_pending)
    }

    /// Stake × time, the reputation signal that self-dealing can't fake.
    fn touch_stake(&mut self, now: i64) {
        let dt = (now - self.stake_touched_at).max(0) as u128;
        let add = self.stake_total as u128 * dt / 86_400;
        self.stake_usd_days = self.stake_usd_days.saturating_add(add as u64);
        self.stake_touched_at = now;
    }

    /// Takes `amount` off the stake. The tokens stay in the seller vault until
    /// the buyer's slash claim moves them, so a frozen vault can't hold up the
    /// refund.
    fn slash(&mut self, amount: u64, now: i64) {
        self.touch_stake(now);
        self.stake_total -= amount;
        self.last_fail_at = now;
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub struct OfferTerms {
    pub kind: u8,
    pub tier: u8,
    pub reach: u8,
    pub price_mode: u8,
    pub hw_class: u16,
    pub mem_gb: u16,
    pub slots_total: u16,
    pub region: u16,
    pub measurement: [u8; 32],
    pub price_a: u64,
    pub price_b: u64,
    pub min_order: u64,
    pub max_order: u64,
    pub bond_bps: u16,
    pub start_window_s: u32,
    pub grace_s: u32,
    pub tokens_per_s: u32,
    pub model_root: [u8; 32],
    pub endpoint: [u8; 64],
    pub valid_until: i64,
}

impl OfferTerms {
    fn check(&self, p: &Params, now: i64) -> Result<()> {
        require!(
            self.slots_total > 0
                && self.price_mode <= PRICE_BY_TIME
                && self.reach <= MAX_REACH
                && self.bond_bps >= p.min_bond_bps
                && self.bond_bps <= MAX_BPS
                && self.start_window_s > 0
                && self.start_window_s <= p.max_start_window
                && self.min_order <= self.max_order
                && self.valid_until > now
                && self.valid_until <= now + MAX_OFFER_TTL
                && (self.kind != KIND_TRAINING
                    || self.price_mode != PRICE_BY_TOKEN
                    || self.tokens_per_s > 0),
            LoraError::BadParams
        );
        Ok(())
    }
}

// Field order is a fixed byte layout: buyers filter offers by memcmp on these
// offsets (common/beetle_chain.py). Changing it means bumping OFFER_SCHEMA.
#[account]
#[derive(InitSpace)]
pub struct Offer {
    pub schema: u8,
    pub status: u8,
    pub kind: u8,
    pub tier: u8,
    /// bit0 PUBLIC_UDP, bit1 ECHO, bit2 RELAY.
    pub reach: u8,
    pub price_mode: u8,
    pub hw_class: u16,
    pub mem_gb: u16,
    pub slots_total: u16,
    pub slots_used: u16,
    pub region: u16,
    pub seller: Pubkey,
    pub measurement: [u8; 32],
    /// Training: per million tokens. Inference: per million input tokens.
    pub price_a: u64,
    /// Training: per hour. Inference: per million output tokens.
    pub price_b: u64,
    pub min_order: u64,
    pub max_order: u64,
    pub bond_bps: u16,
    pub start_window_s: u32,
    /// Training: seconds added to the run budget for setup.
    pub grace_s: u32,
    /// Training: the throughput the run budget is computed from.
    pub tokens_per_s: u32,
    pub model_root: [u8; 32],
    /// Public entry point if it has one: iroh EndpointId (32) || address (19).
    pub endpoint: [u8; 64],
    pub valid_until: i64,
    pub updated_at: i64,
    pub rev: u32,
    pub offer_stake: u64,
    pub idx: u16,
    pub bump: u8,
    /// Jobs delivered but not settled: they hold no slot, but need this account.
    pub settling: u16,
    pub reserved: [u8; 9],
}

impl Offer {
    fn apply(&mut self, t: &OfferTerms, now: i64) {
        self.reach = t.reach;
        self.price_mode = t.price_mode;
        self.hw_class = t.hw_class;
        self.mem_gb = t.mem_gb;
        self.slots_total = t.slots_total;
        self.region = t.region;
        self.measurement = t.measurement;
        self.price_a = t.price_a;
        self.price_b = t.price_b;
        self.min_order = t.min_order;
        self.max_order = t.max_order;
        self.bond_bps = t.bond_bps;
        self.start_window_s = t.start_window_s;
        self.grace_s = t.grace_s;
        self.tokens_per_s = t.tokens_per_s;
        self.model_root = t.model_root;
        self.endpoint = t.endpoint;
        self.valid_until = t.valid_until;
        self.updated_at = now;
    }

    /// (price, run budget in seconds) for `units`. The buyer can't shorten the
    /// run budget and the seller can't stretch it after the order: both come
    /// from the terms the buyer read.
    fn quote(&self, units: u64) -> Result<(u64, i64)> {
        let div_up = |n: u128, d: u128| (n + d - 1) / d;
        let (price, run) = if self.price_mode == PRICE_BY_TOKEN {
            (
                div_up(units as u128 * self.price_a as u128, 1_000_000),
                div_up(units as u128, self.tokens_per_s.max(1) as u128),
            )
        } else {
            (div_up(units as u128 * self.price_b as u128, 3_600), units as u128)
        };
        let price = u64::try_from(price).map_err(|_| LoraError::BadParams)?;
        let run = i64::try_from(run + self.grace_s as u128).map_err(|_| LoraError::BadParams)?;
        Ok((price, run))
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, InitSpace)]
pub enum JobState {
    Opened,
    Started,
    Running,
    Delivered,
    Settled,
    NoStart,
    Rejected,
    NoRelease,
    TimedOut,
    Declined,
    Aborted,
    Unstarted,
}

impl JobState {
    fn is_finished(self) -> bool {
        !matches!(self, JobState::Opened | JobState::Started | JobState::Running | JobState::Delivered)
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum Payee {
    Seller,
    Buyer,
    /// The slashed part of the seller's stake, owed to the buyer.
    Slash,
}

// Field order is the byte layout read by common/beetle_chain.py.
#[account]
#[derive(InitSpace)]
pub struct Job {
    pub requester: Pubkey,
    /// The seller's authority, which is also who gets paid.
    pub host: Pubkey,
    pub payer: Pubkey,
    pub offer: Pubkey,
    /// Optional third party that must sign the registration; default = none.
    pub verifier: Pubkey,
    pub job_id: [u8; 32],
    pub spec_hash: [u8; 32],
    pub price: u64,
    pub bond: u64,
    pub state: JobState,
    pub opened_at: i64,
    pub start_by: i64,
    pub release_by: i64,
    pub run_secs: i64,
    pub deadline: i64,
    pub sign_pk: [u8; 32],
    pub enc_pk: [u8; 32],
    pub measurement: [u8; 32],
    pub evidence_hash: [u8; 32],
    pub has_dek: bool,
    pub sealed_dek: [u8; SEALED_LEN],
    pub outcome: u8,
    pub work_bps: u16,
    pub adapter_ct_hash: [u8; 32],
    pub has_adapter_key: bool,
    pub sealed_adapter_key: [u8; SEALED_LEN],
    pub report_hash: [u8; 32],
    pub challenge_end: i64,
    pub offer_rev: u32,
    pub floor_bps: u16,
    pub fault_bps: u16,
    pub tier: u8,
    pub seller_due: u64,
    pub buyer_due: u64,
    pub slash_due: u64,
    /// Set for good by the first rendezvous. Only clients can see inputs move,
    /// so this is the one sign on chain that the buyer showed up.
    pub rdv_opened: bool,
    /// What was paid for: train tokens or seconds, by `price_mode`.
    pub units: u64,
    /// The part of `run_secs` that was for setup and evaluation.
    pub grace_s: u32,
    pub price_mode: u8,
    /// The seller answered a rendezvous, so it was there.
    pub rdv_answered: bool,
    pub bump: u8,
}

/// One per job. Only the sides of the job can write the blobs, which are sealed
/// to each other: anyone can read this account, and it stays in history forever.
#[account]
#[derive(InitSpace)]
pub struct Rendezvous {
    pub job_id: [u8; 32],
    pub requester: Pubkey,
    pub host: Pubkey,
    pub payer: Pubkey,
    pub opened_at: i64,
    pub expires_at: i64,
    pub policy: u8,
    pub round: u8,
    pub state: u8,
    pub bump: u8,
    pub offer: [u8; BLOB_LEN],
    pub answer: [u8; BLOB_LEN],
}

#[error_code]
pub enum LoraError {
    #[msg("invalid parameters")]
    BadParams,
    #[msg("job id must be SHA256(requester || nonce || open_slot)")]
    BadJobId,
    #[msg("open_slot is not within the recent slot window")]
    StaleOpenSlot,
    #[msg("instruction not allowed in the job's current state")]
    WrongState,
    #[msg("deadline passed")]
    TooLate,
    #[msg("too early")]
    TooEarly,
    #[msg("previous instruction must be an Ed25519 signature check")]
    MissingSigIx,
    #[msg("malformed Ed25519 instruction")]
    BadSigIx,
    #[msg("signature is not from the key this job trusts")]
    WrongSigner,
    #[msg("signed message does not match the submitted result")]
    WrongMessage,
    #[msg("only the requester can do this")]
    NotRequester,
    #[msg("privacy tier not allowed on this network")]
    TierNotAccepted,
    #[msg("verifier registration expired")]
    RegistrationExpired,
    #[msg("only the program's upgrade authority can initialize")]
    NotUpgradeAuthority,
    #[msg("not enough free stake")]
    InsufficientStake,
    #[msg("seller already has the maximum number of offers")]
    TooManyOffers,
    #[msg("offer has jobs in progress")]
    OfferBusy,
    #[msg("offer terms changed since they were read")]
    OfferChanged,
    #[msg("offer is paused or expired")]
    OfferNotActive,
    #[msg("offer has no free slot")]
    NoCapacity,
    #[msg("order size outside the offer's limits")]
    OrderOutOfRange,
    #[msg("price above the buyer's limit")]
    PriceTooHigh,
    #[msg("measurement differs from the offer's")]
    MeasurementMismatch,
    #[msg("evidence hash differs from what the operator committed to")]
    EvidenceMismatch,
    #[msg("the sandbox did not receive the key sealed on chain")]
    SealedDekMismatch,
    #[msg("token account does not belong to this payee")]
    WrongPayee,
    #[msg("nothing is owed to this payee")]
    NothingDue,
    #[msg("only the seller's operator can do this")]
    NotOperator,
    #[msg("only the job's buyer or seller can do this")]
    NotParticipant,
    #[msg("no rendezvous was ever opened for this job")]
    NoRendezvous,
    #[msg("wrong rendezvous round")]
    BadRound,
    #[msg("a proven sandbox signed this result: the buyer can't refuse it")]
    ProvenResult,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offer(price_mode: u8) -> Offer {
        Offer {
            schema: OFFER_SCHEMA, status: OFFER_ACTIVE, kind: KIND_TRAINING, tier: 0, reach: 0,
            price_mode, hw_class: 0, mem_gb: 0, slots_total: 1, slots_used: 0, region: 0,
            seller: Pubkey::default(), measurement: [0; 32], price_a: 1_000_000, price_b: 3_600_000,
            min_order: 0, max_order: u64::MAX, bond_bps: 5_000, start_window_s: 60, grace_s: 30,
            tokens_per_s: 1_000, model_root: [0; 32], endpoint: [0; 64], valid_until: 0,
            updated_at: 0, rev: 0, offer_stake: 0, idx: 0, bump: 0, settling: 0, reserved: [0; 9],
        }
    }

    fn job(outcome: u8, work_bps: u16) -> Job {
        let d = [0u8; 695];
        let mut j = Job::deserialize(&mut &d[..]).unwrap();
        j.price = 1_000_000;
        j.tier = TIER_T2;
        j.floor_bps = 2_000;
        j.fault_bps = 500;
        j.outcome = outcome;
        j.work_bps = work_bps;
        j
    }

    #[test]
    fn account_sizes_match_the_layouts_clients_filter_on() {
        assert_eq!(8 + Offer::INIT_SPACE, 272);
        assert_eq!(8 + Seller::INIT_SPACE, 240);
        assert_eq!(8 + Job::INIT_SPACE, 703);
        assert_eq!(8 + Rendezvous::INIT_SPACE, 604);
        assert_eq!(BLOB_LEN, 224);
    }

    #[test]
    fn settling_takes_bytes_that_were_reserved_and_moves_nothing() {
        let mut o = offer(0);
        (o.bump, o.settling) = (7, 0x0201);
        let mut bytes = Vec::new();
        AnchorSerialize::serialize(&o, &mut bytes).unwrap();
        assert_eq!(bytes.len(), 272 - 8);
        assert_eq!(bytes[252..255], [7, 1, 2]); // bump, then the first reserved bytes
        assert_eq!(bytes[255..], [0; 9]);
    }

    #[test]
    fn chain_tag_is_lowercase_hex() {
        let mut id = [0u8; 32];
        (id[0], id[1], id[31]) = (0x00, 0xab, 0x7f);
        let tag = chain_tag(&id);
        assert_eq!(&tag[..4], b"00ab");
        assert_eq!(&tag[62..], b"7f");
    }

    #[test]
    fn quote_by_token_and_by_time() {
        assert_eq!(offer(PRICE_BY_TOKEN).quote(2_000_000).unwrap(), (2_000_000, 2_030));
        assert_eq!(offer(PRICE_BY_TOKEN).quote(1).unwrap(), (1, 31)); // rounds up
        assert_eq!(offer(PRICE_BY_TIME).quote(1_800).unwrap(), (1_800_000, 1_830));
    }

    #[test]
    fn payouts_follow_the_signed_outcome() {
        assert_eq!(payout(&job(OUTCOME_PASSED, 10_000), false), (1_000_000, 0));
        assert_eq!(payout(&job(OUTCOME_FAILED, 10_000), false), (200_000, 800_000));
        // Attack 1 in the design notes: throttle, stop at 10% of the work.
        assert_eq!(payout(&job(OUTCOME_FAILED, 1_000), false), (20_000, 980_000));
        assert_eq!(payout(&job(OUTCOME_REQUESTER_FAULT, 0), false), (50_000, 950_000));
        assert_eq!(payout(&job(OUTCOME_INCONCLUSIVE, 10_000), false), (0, 1_000_000));
    }

    #[test]
    fn a_buyers_fault_pays_for_the_work_it_used_up() {
        assert_eq!(payout(&job(OUTCOME_REQUESTER_FAULT, 3_000), false), (300_000, 700_000));
        assert_eq!(payout(&job(OUTCOME_REQUESTER_FAULT, 100), false), (50_000, 950_000));
    }

    #[test]
    fn only_a_proven_sandbox_or_the_buyer_releases_the_full_price() {
        for tier in [0, 1, 3, 4] {
            let unproven = |outcome, work| Job { tier, ..job(outcome, work) };
            assert_eq!(payout(&unproven(OUTCOME_PASSED, 10_000), false), (200_000, 800_000));
            assert_eq!(payout(&unproven(OUTCOME_PASSED, 5_000), false), (100_000, 900_000));
            assert_eq!(payout(&unproven(OUTCOME_PASSED, 10_000), true), (1_000_000, 0));
            // A seller can't reach the full price through another outcome.
            assert_eq!(payout(&unproven(OUTCOME_REQUESTER_FAULT, 10_000), false), (50_000, 950_000));
            assert_eq!(payout(&unproven(OUTCOME_FAILED, 10_000), true), (200_000, 800_000));
        }
    }

    #[test]
    fn free_stake_excludes_locked_and_pending() {
        let d = [0u8; 232];
        let mut s = Seller::deserialize(&mut &d[..]).unwrap();
        (s.stake_total, s.stake_locked, s.unstake_pending) = (100, 30, 50);
        assert_eq!(s.free_stake(), 20);
        s.stake_locked = 90; // a slash can only make this tighter, never underflow
        assert_eq!(s.free_stake(), 0);
    }

    #[test]
    fn stake_days_accumulate_with_time() {
        let d = [0u8; 232];
        let mut s = Seller::deserialize(&mut &d[..]).unwrap();
        s.stake_total = 20_000_000;
        s.touch_stake(86_400);
        assert_eq!(s.stake_usd_days, 20_000_000);
        s.touch_stake(86_400 + 43_200);
        assert_eq!(s.stake_usd_days, 30_000_000);
    }
}
