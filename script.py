#!/usr/bin/env python3
"""
PlanetExplorer - NASA Space Apps Challenge 2025
Complete web application generator with enhanced animations
"""


def generate_html():
    """Generate complete HTML with enhanced background animations"""

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlanetExplorer - NASA Space Apps Challenge 2025</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #000000;
            color: #ffffff;
            overflow-x: hidden;
            min-height: 100vh;
            position: relative;
        }

        /* Enhanced animated background */
        .animated-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: 0;
            background: radial-gradient(ellipse at center, #0a0a0a 0%, #000000 70%, #000000 100%);
        }

        /* Rotating galaxy with depth */
        .milky-way {
            position: absolute;
            width: 200%;
            height: 200%;
            top: -50%;
            left: -50%;
            background: 
                radial-gradient(ellipse at 30% 40%, rgba(100, 80, 150, 0.15) 0%, transparent 40%),
                radial-gradient(ellipse at 70% 60%, rgba(80, 100, 200, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(150, 100, 200, 0.08) 0%, transparent 60%);
            animation: rotateMilkyWay 200s linear infinite;
            opacity: 0.6;
        }

        @keyframes rotateMilkyWay {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* Nebula clouds with color variation */
        .nebula {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0;
            animation: nebulaDrift 30s ease-in-out infinite;
        }

        .nebula-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(138, 43, 226, 0.3), transparent);
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }

        .nebula-2 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(0, 191, 255, 0.25), transparent);
            top: 60%;
            right: 15%;
            animation-delay: 10s;
        }

        .nebula-3 {
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, rgba(255, 20, 147, 0.2), transparent);
            bottom: 20%;
            left: 30%;
            animation-delay: 20s;
        }

        @keyframes nebulaDrift {
            0%, 100% {
                opacity: 0.2;
                transform: translate(0, 0) scale(1);
            }
            50% {
                opacity: 0.4;
                transform: translate(100px, -100px) scale(1.2);
            }
        }

        /* Multi-layer stars with different sizes */
        .stars {
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(2px 2px at 20% 30%, white, transparent),
                radial-gradient(2px 2px at 60% 70%, white, transparent),
                radial-gradient(1px 1px at 50% 50%, white, transparent),
                radial-gradient(1px 1px at 80% 10%, white, transparent),
                radial-gradient(2px 2px at 90% 60%, white, transparent),
                radial-gradient(1px 1px at 33% 70%, white, transparent),
                radial-gradient(1px 1px at 79% 80%, white, transparent),
                radial-gradient(3px 3px at 15% 25%, rgba(255,255,255,0.8), transparent);
            background-size: 200% 200%;
            background-position: 0 0;
            animation: twinkle 200s linear infinite;
        }

        .stars-layer2 {
            background-image: 
                radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.8), transparent),
                radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.8), transparent),
                radial-gradient(2px 2px at 70% 40%, rgba(255,255,255,0.9), transparent),
                radial-gradient(2px 2px at 85% 90%, rgba(255,255,255,0.8), transparent),
                radial-gradient(1px 1px at 25% 60%, rgba(255,255,255,0.7), transparent),
                radial-gradient(3px 3px at 45% 35%, rgba(255,255,255,0.6), transparent);
            background-size: 250% 250%;
            animation: twinkle 150s linear infinite reverse;
        }

        .stars-layer3 {
            background-image: 
                radial-gradient(1px 1px at 15% 85%, rgba(0,212,255,0.6), transparent),
                radial-gradient(1px 1px at 55% 25%, rgba(123,97,255,0.6), transparent),
                radial-gradient(2px 2px at 75% 75%, rgba(255,107,157,0.6), transparent),
                radial-gradient(2px 2px at 95% 45%, rgba(255,255,255,0.6), transparent),
                radial-gradient(1px 1px at 35% 55%, rgba(0,255,127,0.5), transparent);
            background-size: 300% 300%;
            animation: twinkle 100s linear infinite;
        }

        @keyframes twinkle {
            0%, 100% { 
                opacity: 0.8;
                background-position: 0 0;
            }
            50% { 
                opacity: 1;
                background-position: 100% 100%;
            }
        }

        /* Cosmic dust particles */
        .cosmic-dust {
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(1px 1px at 25% 45%, rgba(255,255,255,0.1), transparent),
                radial-gradient(1px 1px at 65% 75%, rgba(255,255,255,0.1), transparent),
                radial-gradient(1px 1px at 85% 15%, rgba(255,255,255,0.1), transparent);
            background-size: 400% 400%;
            animation: dustFlow 180s linear infinite;
        }

        @keyframes dustFlow {
            0% { background-position: 0 0; }
            100% { background-position: 100% 100%; }
        }

        /* Solar system animation */
        .solar-system {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 1px;
            height: 1px;
        }

        .sun {
            position: absolute;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, #FDB813, #FF8C00, #FF4500);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 
                0 0 60px #FDB813,
                0 0 100px #FF8C00,
                0 0 140px rgba(255, 140, 0, 0.5);
            animation: sunPulse 4s ease-in-out infinite;
            z-index: 10;
        }

        @keyframes sunPulse {
            0%, 100% {
                transform: translate(-50%, -50%) scale(1);
                box-shadow: 
                    0 0 60px #FDB813,
                    0 0 100px #FF8C00,
                    0 0 140px rgba(255, 140, 0, 0.5);
            }
            50% {
                transform: translate(-50%, -50%) scale(1.1);
                box-shadow: 
                    0 0 80px #FDB813,
                    0 0 120px #FF8C00,
                    0 0 160px rgba(255, 140, 0, 0.6);
            }
        }

        /* Orbital paths */
        .orbit {
            position: absolute;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .orbit-1 { width: 150px; height: 150px; animation: rotateOrbit 20s linear infinite; }
        .orbit-2 { width: 250px; height: 250px; animation: rotateOrbit 35s linear infinite; }
        .orbit-3 { width: 350px; height: 350px; animation: rotateOrbit 50s linear infinite; }
        .orbit-4 { width: 450px; height: 450px; animation: rotateOrbit 70s linear infinite; }
        .orbit-5 { width: 550px; height: 550px; animation: rotateOrbit 90s linear infinite; }

        @keyframes rotateOrbit {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }

        /* Planets */
        .planet {
            position: absolute;
            border-radius: 50%;
            top: 50%;
            left: 50%;
        }

        .mercury {
            width: 12px;
            height: 12px;
            background: radial-gradient(circle at 30% 30%, #C0C0C0, #808080);
            transform: translate(-50%, -50%) translateX(75px);
            box-shadow: 0 0 10px rgba(192, 192, 192, 0.5);
            animation: orbitMercury 20s linear infinite;
        }

        @keyframes orbitMercury {
            from { transform: translate(-50%, -50%) rotate(0deg) translateX(75px) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg) translateX(75px) rotate(-360deg); }
        }

        .venus {
            width: 18px;
            height: 18px;
            background: radial-gradient(circle at 30% 30%, #FFC649, #CD853F);
            transform: translate(-50%, -50%) translateX(125px);
            box-shadow: 0 0 15px rgba(255, 198, 73, 0.6);
            animation: orbitVenus 35s linear infinite;
        }

        @keyframes orbitVenus {
            from { transform: translate(-50%, -50%) rotate(0deg) translateX(125px) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg) translateX(125px) rotate(-360deg); }
        }

        .earth {
            width: 20px;
            height: 20px;
            background: radial-gradient(circle at 30% 30%, #4A90E2, #1E5BA8, #0A2E5C);
            transform: translate(-50%, -50%) translateX(175px);
            box-shadow: 0 0 20px rgba(74, 144, 226, 0.7);
            animation: orbitEarth 50s linear infinite;
        }

        @keyframes orbitEarth {
            from { transform: translate(-50%, -50%) rotate(0deg) translateX(175px) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg) translateX(175px) rotate(-360deg); }
        }

        .mars {
            width: 16px;
            height: 16px;
            background: radial-gradient(circle at 30% 30%, #E27B58, #C1440E);
            transform: translate(-50%, -50%) translateX(225px);
            box-shadow: 0 0 15px rgba(226, 123, 88, 0.6);
            animation: orbitMars 70s linear infinite;
        }

        @keyframes orbitMars {
            from { transform: translate(-50%, -50%) rotate(0deg) translateX(225px) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg) translateX(225px) rotate(-360deg); }
        }

        .jupiter {
            width: 35px;
            height: 35px;
            background: radial-gradient(circle at 30% 30%, #C88B3A, #8B4513);
            transform: translate(-50%, -50%) translateX(275px);
            box-shadow: 0 0 25px rgba(200, 139, 58, 0.6);
            animation: orbitJupiter 90s linear infinite;
        }

        @keyframes orbitJupiter {
            from { transform: translate(-50%, -50%) rotate(0deg) translateX(275px) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg) translateX(275px) rotate(-360deg); }
        }

        /* Comet trails */
        .comet {
            position: absolute;
            width: 4px;
            height: 4px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 0 20px white, -100px 0 30px rgba(255,255,255,0.3);
            animation: cometFly 15s linear infinite;
            opacity: 0;
        }

        @keyframes cometFly {
            0% {
                transform: translate(100vw, -100px) rotate(-45deg);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translate(-100px, 100vh) rotate(-45deg);
                opacity: 0;
            }
        }

        .comet:nth-child(1) { top: 20%; animation-delay: 3s; }
        .comet:nth-child(2) { top: 50%; animation-delay: 8s; }
        .comet:nth-child(3) { top: 70%; animation-delay: 13s; }

        /* Shooting stars */
        .shooting-star {
            position: absolute;
            width: 2px;
            height: 2px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 0 10px 2px white;
            animation: shoot 3s linear infinite;
            opacity: 0;
        }

        @keyframes shoot {
            0% {
                transform: translateX(0) translateY(0);
                opacity: 1;
            }
            100% {
                transform: translateX(-300px) translateY(300px);
                opacity: 0;
            }
        }

        .shooting-star:nth-child(1) { top: 10%; left: 80%; animation-delay: 2s; }
        .shooting-star:nth-child(2) { top: 30%; left: 90%; animation-delay: 5s; }
        .shooting-star:nth-child(3) { top: 50%; left: 85%; animation-delay: 8s; }
        .shooting-star:nth-child(4) { top: 70%; left: 95%; animation-delay: 11s; }

        /* Floating particles */
        .particle {
            position: absolute;
            border-radius: 50%;
            animation: float linear infinite;
            opacity: 0.5;
            box-shadow: 0 0 20px currentColor;
        }

        @keyframes float {
            0% {
                transform: translateY(100vh) translateX(0) scale(0) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 0.6;
            }
            90% {
                opacity: 0.6;
            }
            100% {
                transform: translateY(-100vh) translateX(100px) scale(1.5) rotate(360deg);
                opacity: 0;
            }
        }

        /* Energy waves */
        .energy-wave {
            position: absolute;
            width: 300px;
            height: 300px;
            border: 2px solid rgba(0, 212, 255, 0.1);
            border-radius: 50%;
            top: 30%;
            left: 20%;
            animation: energyPulse 8s ease-in-out infinite;
        }

        @keyframes energyPulse {
            0%, 100% {
                transform: scale(0.8);
                opacity: 0;
            }
            50% {
                transform: scale(2);
                opacity: 0.3;
            }
        }

        .energy-wave:nth-child(2) {
            top: 60%;
            left: 70%;
            animation-delay: 4s;
            border-color: rgba(123, 97, 255, 0.1);
        }

        .container {
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 24px 150px;
        }

        .section {
            display: none;
        }

        .section.active {
            display: block;
        }

        .hero {
            margin-bottom: 48px;
            text-align: center;
        }

        .hero-label {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 3px;
            color: #00D4FF;
            margin-bottom: 16px;
            text-transform: uppercase;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
            animation: glow 2s ease-in-out infinite;
        }

        @keyframes glow {
            0%, 100% {
                text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
            }
            50% {
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.7), 0 0 60px rgba(0, 212, 255, 0.3);
            }
        }

        .hero-title {
            font-size: 56px;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff, #00D4FF, #7B61FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 16px;
            animation: titleShine 3s ease-in-out infinite;
        }

        @keyframes titleShine {
            0%, 100% {
                filter: brightness(1) drop-shadow(0 0 20px rgba(0, 212, 255, 0.2));
            }
            50% {
                filter: brightness(1.3) drop-shadow(0 0 40px rgba(0, 212, 255, 0.4));
            }
        }

        .hero-subtitle {
            font-size: 20px;
            color: #A0A0A0;
            line-height: 1.6;
            max-width: 700px;
            margin: 0 auto;
        }

        .glass-card {
            position: relative;
            background: rgba(20, 20, 20, 0.4);
            backdrop-filter: blur(40px) saturate(150%);
            -webkit-backdrop-filter: blur(40px) saturate(150%);
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 32px;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.6),
                0 0 0 1px rgba(255, 255, 255, 0.05) inset,
                0 0 60px rgba(0, 212, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.03), transparent);
            transform: rotate(45deg);
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0% {
                transform: translateX(-100%) translateY(-100%) rotate(45deg);
            }
            100% {
                transform: translateX(100%) translateY(100%) rotate(45deg);
            }
        }

        .glass-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 
                0 16px 48px rgba(0, 0, 0, 0.7),
                0 0 0 1px rgba(255, 255, 255, 0.1) inset,
                0 0 100px rgba(0, 212, 255, 0.15);
            background: rgba(25, 25, 25, 0.5);
            border-color: rgba(0, 212, 255, 0.3);
        }

        .main-card {
            text-align: center;
            background: rgba(15, 15, 15, 0.5);
            backdrop-filter: blur(50px) saturate(180%);
        }

        .sparkles-icon {
            font-size: 64px;
            background: linear-gradient(135deg, #00D4FF, #7B61FF, #FF6B9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 24px;
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.4));
            animation: bounce 2s ease-in-out infinite;
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0) scale(1);
            }
            50% {
                transform: translateY(-10px) scale(1.05);
            }
        }

        .main-card h2 {
            font-size: 36px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 16px;
        }

        .main-card p {
            font-size: 17px;
            color: #A0A0A0;
            line-height: 1.7;
            margin-bottom: 32px;
            max-width: 650px;
            margin-left: auto;
            margin-right: auto;
        }

        .cta-button {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            background: linear-gradient(135deg, #00D4FF, #7B61FF);
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            padding: 18px 36px;
            border-radius: 16px;
            border: none;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .cta-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent);
            transition: left 0.5s;
        }

        .cta-button:hover::before {
            left: 100%;
        }

        .cta-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0, 212, 255, 0.5);
            background: linear-gradient(135deg, #00E4FF, #8B71FF);
        }

        .cta-button:active {
            transform: translateY(-1px);
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-top: 48px;
        }

        .feature-card {
            position: relative;
            background: rgba(20, 20, 20, 0.35);
            backdrop-filter: blur(40px) saturate(150%);
            -webkit-backdrop-filter: blur(40px) saturate(150%);
            border-radius: 20px;
            padding: 32px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.02), transparent);
            transform: rotate(45deg);
            animation: shimmer 4s infinite;
        }

        .feature-card:hover {
            transform: translateY(-8px);
            background: rgba(25, 25, 25, 0.45);
            box-shadow: 
                0 16px 48px rgba(0, 212, 255, 0.15),
                0 0 0 1px rgba(0, 212, 255, 0.2) inset;
            border-color: rgba(0, 212, 255, 0.3);
        }

        .feature-icon {
            font-size: 48px;
            margin-bottom: 20px;
            filter: drop-shadow(0 0 15px rgba(0, 212, 255, 0.3));
            animation: iconFloat 3s ease-in-out infinite;
        }

        @keyframes iconFloat {
            0%, 100% {
                transform: translateY(0) rotate(0deg);
            }
            50% {
                transform: translateY(-5px) rotate(5deg);
            }
        }

        .feature-card h3 {
            font-size: 22px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 12px;
        }

        .feature-card p {
            font-size: 15px;
            color: #A0A0A0;
            line-height: 1.6;
        }

        .info-section {
            margin-top: 48px;
        }

        .info-section h3 {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 20px;
            text-align: center;
        }

        .info-section p {
            font-size: 16px;
            color: #A0A0A0;
            line-height: 1.7;
            text-align: center;
            max-width: 800px;
            margin: 0 auto 32px;
        }

        .upload-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(15px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease;
        }

        .upload-modal.active {
            display: flex;
        }

        .upload-modal-content {
            background: rgba(15, 15, 15, 0.7);
            backdrop-filter: blur(50px) saturate(180%);
            border-radius: 24px;
            padding: 48px;
            max-width: 500px;
            width: 90%;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.8),
                0 0 0 1px rgba(255, 255, 255, 0.08) inset,
                0 0 80px rgba(0, 212, 255, 0.15);
            position: relative;
            animation: slideUp 0.4s ease;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .upload-modal-close {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 20px;
            transition: all 0.3s ease;
        }

        .upload-modal-close:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: rotate(90deg);
        }

        .upload-modal-icon {
            font-size: 64px;
            text-align: center;
            margin-bottom: 24px;
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.4));
        }

        .upload-modal h2 {
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            margin-bottom: 16px;
        }

        .upload-modal p {
            font-size: 16px;
            color: #A0A0A0;
            text-align: center;
            margin-bottom: 32px;
            line-height: 1.6;
        }

        .file-upload-area {
            border: 2px dashed rgba(0, 212, 255, 0.25);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            background: rgba(0, 212, 255, 0.03);
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 24px;
        }

        .file-upload-area:hover {
            border-color: rgba(0, 212, 255, 0.5);
            background: rgba(0, 212, 255, 0.08);
            transform: scale(1.02);
        }

        .file-upload-area input[type="file"] {
            display: none;
        }

        .file-upload-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }

        .file-upload-text {
            font-size: 16px;
            color: #ffffff;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .file-upload-subtext {
            font-size: 14px;
            color: #808080;
        }

        .format-badges {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .badge {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            color: #00D4FF;
        }

        .selected-file {
            text-align: center;
            padding: 16px;
            background: rgba(0, 212, 255, 0.08);
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 12px;
            margin-top: 16px;
            display: none;
        }

        .selected-file.active {
            display: block;
        }

        .selected-file-icon {
            font-size: 24px;
            margin-bottom: 8px;
        }

        .selected-file-name {
            font-size: 14px;
            color: #00D4FF;
            font-weight: 600;
        }

        .loading-screen {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.98);
            backdrop-filter: blur(20px);
            z-index: 2000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .loading-screen.active {
            display: flex;
        }

        .loading-content {
            text-align: center;
            max-width: 600px;
            padding: 40px;
        }

        .telescope-animation {
            font-size: 120px;
            margin-bottom: 40px;
            animation: telescopeScan 3s ease-in-out infinite;
            filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.6));
        }

        @keyframes telescopeScan {
            0%, 100% {
                transform: rotate(-10deg) scale(1);
            }
            25% {
                transform: rotate(10deg) scale(1.1);
            }
            50% {
                transform: rotate(-10deg) scale(1);
            }
            75% {
                transform: rotate(10deg) scale(1.1);
            }
        }

        .loading-spinner-container {
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0 auto 40px;
        }

        .spinner-ring {
            position: absolute;
            border-radius: 50%;
            border: 3px solid transparent;
            animation: spinRing 2s linear infinite;
        }

        .spinner-ring-1 {
            width: 150px;
            height: 150px;
            border-top-color: #00D4FF;
            border-right-color: #00D4FF;
            animation-duration: 1.5s;
        }

        .spinner-ring-2 {
            width: 120px;
            height: 120px;
            border-bottom-color: #7B61FF;
            border-left-color: #7B61FF;
            top: 15px;
            left: 15px;
            animation-duration: 2s;
            animation-direction: reverse;
        }

        .spinner-ring-3 {
            width: 90px;
            height: 90px;
            border-top-color: #FF6B9D;
            border-right-color: #FF6B9D;
            top: 30px;
            left: 30px;
            animation-duration: 1s;
        }

        @keyframes spinRing {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-orb {
            position: absolute;
            width: 20px;
            height: 20px;
            background: radial-gradient(circle, #00D4FF, #7B61FF);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 30px #00D4FF;
            animation: orbPulse 1.5s ease-in-out infinite;
        }

        @keyframes orbPulse {
            0%, 100% {
                transform: translate(-50%, -50%) scale(1);
                box-shadow: 0 0 30px #00D4FF;
            }
            50% {
                transform: translate(-50%, -50%) scale(1.3);
                box-shadow: 0 0 50px #00D4FF, 0 0 80px #7B61FF;
            }
        }

        .loading-text {
            font-size: 28px;
            color: #ffffff;
            font-weight: 700;
            text-align: center;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #00D4FF, #7B61FF, #FF6B9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textShimmer 2s ease-in-out infinite;
        }

        @keyframes textShimmer {
            0%, 100% {
                filter: brightness(1);
            }
            50% {
                filter: brightness(1.5);
            }
        }

        .loading-stage {
            font-size: 16px;
            color: #00D4FF;
            font-weight: 600;
            margin-bottom: 12px;
            animation: fadeInOut 2s ease-in-out infinite;
        }

        @keyframes fadeInOut {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        .loading-subtext {
            font-size: 14px;
            color: #808080;
            line-height: 1.6;
        }

        .progress-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            overflow: hidden;
            margin: 24px 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00D4FF, #7B61FF, #FF6B9D);
            border-radius: 10px;
            animation: progressAnimation 3s ease-in-out infinite;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }

        @keyframes progressAnimation {
            0% {
                width: 0%;
            }
            50% {
                width: 70%;
            }
            100% {
                width: 95%;
            }
        }

        .analysis-icons {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 30px;
        }

        .analysis-icon {
            font-size: 40px;
            opacity: 0;
            animation: iconFadeIn 3s ease-in-out infinite;
        }

        .analysis-icon:nth-child(1) { animation-delay: 0s; }
        .analysis-icon:nth-child(2) { animation-delay: 0.5s; }
        .analysis-icon:nth-child(3) { animation-delay: 1s; }
        .analysis-icon:nth-child(4) { animation-delay: 1.5s; }

        @keyframes iconFadeIn {
            0%, 100% {
                opacity: 0;
                transform: translateY(20px);
            }
            30%, 70% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .results-header {
            text-align: center;
            margin-bottom: 48px;
        }

        .results-header h2 {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4FF, #7B61FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .results-header .filename {
            font-size: 16px;
            color: #808080;
            font-weight: 500;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 16px;
        }

        .status-badge.detected {
            background: rgba(46, 204, 113, 0.15);
            border: 1px solid rgba(46, 204, 113, 0.3);
            color: #2ECC71;
        }

        .status-badge.not-detected {
            background: rgba(231, 76, 60, 0.15);
            border: 1px solid rgba(231, 76, 60, 0.3);
            color: #E74C3C;
        }

        .confidence-card {
            text-align: center;
            padding: 40px;
        }

        .confidence-circle {
            width: 200px;
            height: 200px;
            margin: 0 auto 24px;
            position: relative;
        }

        .confidence-circle svg {
            transform: rotate(-90deg);
        }

        .confidence-circle circle {
            fill: none;
            stroke-width: 12;
        }

        .confidence-bg {
            stroke: rgba(255, 255, 255, 0.05);
        }

        .confidence-progress {
            stroke: url(#confidenceGradient);
            stroke-linecap: round;
            transition: stroke-dashoffset 2s ease;
        }

        .confidence-value {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4FF, #7B61FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .confidence-label {
            font-size: 18px;
            color: #A0A0A0;
            font-weight: 600;
        }

        .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-top: 48px;
        }

        .param-card {
            background: rgba(15, 15, 15, 0.5);
            backdrop-filter: blur(40px);
            border-radius: 20px;
            padding: 28px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
        }

        .param-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 212, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        }

        .param-icon {
            font-size: 36px;
            margin-bottom: 12px;
            filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));
        }

        .param-label {
            font-size: 13px;
            color: #808080;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .param-value {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }

        .param-unit {
            font-size: 14px;
            color: #A0A0A0;
        }

        .chart-card {
            margin-top: 48px;
            padding: 40px;
        }

        .chart-card h3 {
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 24px;
            text-align: center;
        }

        .chart-placeholder {
            width: 100%;
            height: 400px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            border: 2px dashed rgba(255, 255, 255, 0.1);
        }

        .chart-placeholder-icon {
            font-size: 64px;
            margin-bottom: 16px;
            opacity: 0.3;
        }

        .chart-placeholder-text {
            font-size: 16px;
            color: #808080;
        }

        .results-actions {
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-top: 48px;
            flex-wrap: wrap;
        }

        .action-btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 16px 32px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }

        .btn-primary {
            background: linear-gradient(135deg, #00D4FF, #7B61FF);
            color: #ffffff;
            box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(0, 212, 255, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .raw-data-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            backdrop-filter: blur(20px);
            z-index: 3000;
            align-items: center;
            justify-content: center;
        }

        .raw-data-modal.active {
            display: flex;
        }

        .raw-data-content {
            background: rgba(15, 15, 15, 0.9);
            backdrop-filter: blur(50px);
            border-radius: 24px;
            padding: 40px;
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
        }

        .raw-data-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }

        .raw-data-header h3 {
            font-size: 24px;
            color: #ffffff;
        }

        .raw-data-close {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 20px;
            transition: all 0.3s ease;
        }

        .raw-data-close:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: rotate(90deg);
        }

        .json-display {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #00D4FF;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .json-key {
            color: #FF6B9D;
        }

        .json-string {
            color: #2ECC71;
        }

        .json-number {
            color: #FFD700;
        }

        .json-boolean {
            color: #7B61FF;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 40px;
            }

            .hero-subtitle {
                font-size: 17px;
            }

            .main-card h2 {
                font-size: 28px;
            }

            .glass-card {
                padding: 28px;
            }

            .feature-grid {
                grid-template-columns: 1fr;
            }

            .upload-modal-content {
                padding: 32px 24px;
            }
            
            .sun {
                width: 50px;
                height: 50px;
            }
            
            .orbit-1 { width: 100px; height: 100px; }
            .orbit-2 { width: 180px; height: 180px; }
            .orbit-3 { width: 260px; height: 260px; }
            .orbit-4 { width: 340px; height: 340px; }
            .orbit-5 { width: 420px; height: 420px; }
            
            .jupiter { width: 25px; height: 25px; }

            .results-header h2 {
                font-size: 32px;
            }

            .confidence-circle {
                width: 150px;
                height: 150px;
            }

            .confidence-value {
                font-size: 36px;
            }

            .params-grid {
                grid-template-columns: 1fr;
            }

            .results-actions {
                flex-direction: column;
            }

            .action-btn {
                width: 100%;
                justify-content: center;
            }

            .telescope-animation {
                font-size: 80px;
            }

            .loading-spinner-container {
                width: 120px;
                height: 120px;
            }

            .spinner-ring-1 {
                width: 120px;
                height: 120px;
            }

            .spinner-ring-2 {
                width: 90px;
                height: 90px;
            }

            .spinner-ring-3 {
                width: 60px;
                height: 60px;
            }

            .loading-text {
                font-size: 22px;
            }

            .analysis-icons {
                gap: 20px;
            }

            .analysis-icon {
                font-size: 30px;
            }

            .raw-data-content {
                padding: 24px;
            }

            .nebula {
                filter: blur(60px);
            }

            .energy-wave {
                width: 200px;
                height: 200px;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="animated-background" id="animatedBg">
        <div class="milky-way"></div>
        <div class="nebula nebula-1"></div>
        <div class="nebula nebula-2"></div>
        <div class="nebula nebula-3"></div>
        <div class="cosmic-dust"></div>
        <div class="stars"></div>
        <div class="stars stars-layer2"></div>
        <div class="stars stars-layer3"></div>
        <div class="energy-wave"></div>
        <div class="energy-wave"></div>
        <div class="comet"></div>
        <div class="comet"></div>
        <div class="comet"></div>
        <div class="shooting-star"></div>
        <div class="shooting-star"></div>
        <div class="shooting-star"></div>
        <div class="shooting-star"></div>
        <div class="solar-system">
            <div class="sun"></div>
            <div class="orbit orbit-1"></div>
            <div class="orbit orbit-2"></div>
            <div class="orbit orbit-3"></div>
            <div class="orbit orbit-4"></div>
            <div class="orbit orbit-5"></div>
            <div class="planet mercury"></div>
            <div class="planet venus"></div>
            <div class="planet earth"></div>
            <div class="planet mars"></div>
            <div class="planet jupiter"></div>
        </div>
    </div>

    <div class="section active" id="homeSection">
        <div class="container">
            <div class="hero">
                <div class="hero-label">NASA SPACE APPS CHALLENGE 2025</div>
                <h1 class="hero-title">PlanetExplorer</h1>
                <p class="hero-subtitle">Analyze light curves from NASA's Kepler and TESS missions to discover exoplanets using advanced AI</p>
            </div>

            <div class="glass-card main-card">
                <div class="sparkles-icon">✨</div>
                <h2>Start Your Journey</h2>
                <p>Using NASA's open data from Kepler and TESS missions, this AI-powered tool helps identify potential exoplanets through transit method analysis.</p>
                <button class="cta-button" onclick="openUploadModal()">
                    <span>🚀</span>
                    <span>Begin Exploring</span>
                </button>
            </div>

            <div class="feature-grid">
                <div class="feature-card glass-card">
                    <div class="feature-icon">🔭</div>
                    <h3>Real NASA Data</h3>
                    <p>Access authentic light curve data from Kepler and TESS missions, the same data used by astronomers worldwide.</p>
                </div>

                <div class="feature-card glass-card">
                    <div class="feature-icon">🤖</div>
                    <h3>AI-Powered Detection</h3>
                    <p>Advanced machine learning algorithms analyze transit patterns to identify potential exoplanet candidates.</p>
                </div>

                <div class="feature-card glass-card">
                    <div class="feature-icon">📊</div>
                    <h3>Interactive Analysis</h3>
                    <p>Visualize light curves, explore transit depths, and understand the science behind exoplanet discovery.</p>
                </div>
            </div>

            <div class="info-section glass-card">
                <h3>About the Transit Method</h3>
                <p>When a planet passes in front of its host star, it blocks a small fraction of the star's light, creating a characteristic dip in brightness. By analyzing these periodic dips in light curves, we can detect and characterize exoplanets millions of light-years away.</p>
            </div>
        </div>
    </div>

    <div class="section" id="resultsSection">
        <div class="container">
            <div class="results-header">
                <h2>Analysis Results</h2>
                <p class="filename" id="resultsFilename">kepler_data.csv</p>
                <div class="status-badge detected" id="statusBadge">
                    <span>✓</span>
                    <span id="statusText">Exoplanet Detected</span>
                </div>
            </div>

            <div class="glass-card confidence-card">
                <div class="confidence-circle">
                    <svg width="200" height="200">
                        <defs>
                            <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#00D4FF;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#7B61FF;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                        <circle class="confidence-bg" cx="100" cy="100" r="90"></circle>
                        <circle class="confidence-progress" cx="100" cy="100" r="90" 
                                stroke-dasharray="565.48" stroke-dashoffset="113" id="confidenceCircle"></circle>
                    </svg>
                    <div class="confidence-value" id="confidenceValue">95%</div>
                </div>
                <div class="confidence-label">Detection Confidence</div>
            </div>

            <div class="params-grid">
                <div class="param-card glass-card">
                    <div class="param-icon">🌍</div>
                    <div class="param-label">Planet Radius</div>
                    <div class="param-value" id="paramRadius">1.2</div>
                    <div class="param-unit">Earth Radii (R⊕)</div>
                </div>

                <div class="param-card glass-card">
                    <div class="param-icon">🔄</div>
                    <div class="param-label">Orbital Period</div>
                    <div class="param-value" id="paramPeriod">12.5</div>
                    <div class="param-unit">Days</div>
                </div>

                <div class="param-card glass-card">
                    <div class="param-icon">🌡️</div>
                    <div class="param-label">Equilibrium Temp</div>
                    <div class="param-value" id="paramTemp">450</div>
                    <div class="param-unit">Kelvin</div>
                </div>

                <div class="param-card glass-card">
                    <div class="param-icon">📏</div>
                    <div class="param-label">Transit Depth</div>
                    <div class="param-value" id="paramDepth">0.8</div>
                    <div class="param-unit">Percent (%)</div>
                </div>

                <div class="param-card glass-card">
                    <div class="param-icon">⏱️</div>
                    <div class="param-label">Transit Duration</div>
                    <div class="param-value" id="paramDuration">3.2</div>
                    <div class="param-unit">Hours</div>
                </div>

                <div class="param-card glass-card">
                    <div class="param-icon">⭐</div>
                    <div class="param-label">Host Star Type</div>
                    <div class="param-value" id="paramStar" style="font-size: 24px;">G2V</div>
                    <div class="param-unit">Spectral Class</div>
                </div>
            </div>

            <div class="glass-card chart-card">
                <h3>📈 Light Curve Visualization</h3>
                <div class="chart-placeholder">
                    <div class="chart-placeholder-icon">📊</div>
                    <div class="chart-placeholder-text">
                        Light curve chart will be displayed here<br>
                        <span style="font-size: 12px; color: #606060;">(Integrate with Plotly, Chart.js, or D3.js)</span>
                    </div>
                </div>
            </div>

            <div class="results-actions">
                <button class="action-btn btn-primary" onclick="analyzeAnother()">
                    <span>🔄</span>
                    <span>Analyze Another</span>
                </button>
                <button class="action-btn btn-secondary" onclick="downloadReport()">
                    <span>📥</span>
                    <span>Download Report</span>
                </button>
                <button class="action-btn btn-secondary" onclick="viewRawData()">
                    <span>📋</span>
                    <span>View Raw Data</span>
                </button>
            </div>
        </div>
    </div>

    <div class="upload-modal" id="uploadModal">
        <div class="upload-modal-content">
            <div class="upload-modal-close" onclick="closeUploadModal()">✕</div>
            <div class="upload-modal-icon">📁</div>
            <h2>Upload Dataset</h2>
            <p>Select your light curve data file to begin analysis</p>

            <div class="file-upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept=".csv,.xlsx" onchange="handleFileSelect(event)">
                <div class="file-upload-icon">📄</div>
                <div class="file-upload-text">Click to browse files</div>
                <div class="file-upload-subtext">or drag and drop here</div>
            </div>

            <div class="selected-file" id="selectedFile">
                <div class="selected-file-icon">✅</div>
                <div class="selected-file-name" id="selectedFileName"></div>
            </div>

            <div class="format-badges">
                <span class="badge">CSV</span>
                <span class="badge">XLSX</span>
            </div>
        </div>
    </div>

    <div class="loading-screen" id="loadingScreen">
        <div class="loading-content">
            <div class="telescope-animation">🔭</div>
            
            <div class="loading-spinner-container">
                <div class="spinner-ring spinner-ring-1"></div>
                <div class="spinner-ring spinner-ring-2"></div>
                <div class="spinner-ring spinner-ring-3"></div>
                <div class="loading-orb"></div>
            </div>

            <div class="loading-text">Analyzing Light Curves</div>
            <div class="loading-stage" id="loadingStage">Initializing AI models...</div>
            
            <div class="progress-container">
                <div class="progress-bar"></div>
            </div>

            <div class="loading-subtext">
                Scanning for transit signals and anomalies<br>
                This may take a few moments...
            </div>

            <div class="analysis-icons">
                <span class="analysis-icon">📊</span>
                <span class="analysis-icon">🤖</span>
                <span class="analysis-icon">🔬</span>
                <span class="analysis-icon">✨</span>
            </div>
        </div>
    </div>

    <div class="raw-data-modal" id="rawDataModal">
        <div class="raw-data-content">
            <div class="raw-data-header">
                <h3>📋 Raw Analysis Data</h3>
                <div class="raw-data-close" onclick="closeRawDataModal()">✕</div>
            </div>
            <div class="json-display" id="jsonDisplay"></div>
        </div>
    </div>

    <script>
        let currentAnalysisData = null;
        let currentFileName = '';
        const loadingStages = [
            "Initializing AI models...",
            "Processing light curve data...",
            "Detecting transit signals...",
            "Analyzing flux variations...",
            "Calculating orbital parameters...",
            "Finalizing results..."
        ];
        let currentStageIndex = 0;

        // Generate cosmic particles dynamically
        function createParticles() {
            const background = document.getElementById('animatedBg');
            const particleCount = 50;

            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                const size = Math.random() * 3 + 1;
                const startX = Math.random() * window.innerWidth;
                const duration = Math.random() * 25 + 15;
                const delay = Math.random() * 15;
                const colors = ['0, 212, 255', '123, 97, 255', '255, 107, 157'];
                const randomColor = colors[Math.floor(Math.random() * colors.length)];

                particle.style.width = size + 'px';
                particle.style.height = size + 'px';
                particle.style.left = startX + 'px';
                particle.style.background = `rgba(\${randomColor}, 0.6)`;
                particle.style.color = `rgba(\${randomColor}, 0.8)`;
                particle.style.animationDuration = duration + 's';
                particle.style.animationDelay = delay + 's';
                background.appendChild(particle);
            }
        }

        function openUploadModal() {
            document.getElementById('uploadModal').classList.add('active');
        }

        function closeUploadModal() {
            document.getElementById('uploadModal').classList.remove('active');
            document.getElementById('selectedFile').classList.remove('active');
            document.getElementById('fileInput').value = '';
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                const fileName = file.name;
                const fileSize = (file.size / 1024).toFixed(2) + ' KB';
                document.getElementById('selectedFileName').textContent = `${fileName} (${fileSize})`;

                document.getElementById('selectedFile').classList.add('active');
                setTimeout(() => {
                    processFile(file, fileName);
                }, 800);
            }
        }

        // Backend integration point - replace with actual API call
        async function processFile(file, fileName) {
            closeUploadModal();
            showLoading();
            currentFileName = fileName;

            // Replace this setTimeout with your actual API call:
            // const formData = new FormData();
            // formData.append('file', file);
            // const response = await fetch('YOUR_API_ENDPOINT', { method: 'POST', body: formData });
            // const jsonData = await response.json();
            // displayResults(jsonData, fileName);

            setTimeout(() => {
                const mockData = {
                    exoplanet_detected: true,
                    confidence: 0.95,
                    planet_radius: 1.2,
                    orbital_period: 12.5,
                    equilibrium_temp: 450,
                    transit_depth: 0.8,
                    transit_duration: 3.2,
                    host_star_type: "G2V",
                    timestamp: new Date().toISOString(),
                    analysis_time: "2.4 seconds"
                };
                displayResults(mockData, fileName);
            }, 8000);
        }
        
        function displayResults(data, fileName) {
            currentAnalysisData = data;
            hideLoading();

            document.getElementById('resultsFilename').textContent = fileName;

            const statusBadge = document.getElementById('statusBadge');
            const statusText = document.getElementById('statusText');
            
            if (data.exoplanet_detected) {
                statusBadge.className = 'status-badge detected';
                statusText.textContent = 'Exoplanet Detected';
            } else {
                statusBadge.className = 'status-badge not-detected';
                statusText.textContent = 'No Exoplanet Detected';
            }

            const confidence = Math.round(data.confidence * 100);
            document.getElementById('confidenceValue').textContent = confidence + '%';
            
            const circumference = 2 * Math.PI * 90;
            const offset = circumference - (confidence / 100) * circumference;
            document.getElementById('confidenceCircle').style.strokeDashoffset = offset;

            document.getElementById('paramRadius').textContent = data.planet_radius.toFixed(2);
            document.getElementById('paramPeriod').textContent = data.orbital_period.toFixed(1);
            document.getElementById('paramTemp').textContent = Math.round(data.equilibrium_temp);
            document.getElementById('paramDepth').textContent = data.transit_depth.toFixed(2);
            document.getElementById('paramDuration').textContent = data.transit_duration.toFixed(1);
            document.getElementById('paramStar').textContent = data.host_star_type;

            showSection('resultsSection');
        }

        function showLoading() {
            document.getElementById('loadingScreen').classList.add('active');
            currentStageIndex = 0;
            updateLoadingStage();
        }

        function hideLoading() {
            document.getElementById('loadingScreen').classList.remove('active');
        }

        function updateLoadingStage() {
            const stageElement = document.getElementById('loadingStage');
            if (currentStageIndex < loadingStages.length) {
                stageElement.textContent = loadingStages[currentStageIndex];
                currentStageIndex++;
                setTimeout(updateLoadingStage, 1200);
            }
        }

        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            document.getElementById(sectionId).classList.add('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function analyzeAnother() {
            showSection('homeSection');
        }

        function downloadReport() {
            if (!currentAnalysisData) {
                alert('No analysis data available');
                return;
            }

            const reportContent = `
PLANETEXPLORER - EXOPLANET ANALYSIS REPORT
============================================
Generated: $${new Date().toLocaleString()}
File: $${currentFileName}

DETECTION STATUS
================
Exoplanet Detected: $${currentAnalysisData.exoplanet_detected ? 'YES' : 'NO'}
Confidence: $${(currentAnalysisData.confidence * 100).toFixed(2)}%

PLANET PARAMETERS
==================
Planet Radius: $${currentAnalysisData.planet_radius.toFixed(2)} Earth Radii (R⊕)
Orbital Period: $${currentAnalysisData.orbital_period.toFixed(1)} Days
Equilibrium Temperature: $${Math.round(currentAnalysisData.equilibrium_temp)} Kelvin
Transit Depth: $${currentAnalysisData.transit_depth.toFixed(2)}%
Transit Duration: $${currentAnalysisData.transit_duration.toFixed(1)} Hours
Host Star Type: $${currentAnalysisData.host_star_type}

ADDITIONAL INFORMATION
======================
Analysis Time: $${currentAnalysisData.analysis_time || 'N/A'}
Timestamp: $${currentAnalysisData.timestamp || 'N/A'}

============================================
Report generated by PlanetExplorer
NASA Space Apps Challenge 2025
============================================
`;

            const blob = new Blob([reportContent], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `planetexplorer_report_$${currentFileName.replace(/\\.[^/.]+$/, "")}_$${Date.now()}.txt`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }

        function viewRawData() {
            if (!currentAnalysisData) {
                alert('No analysis data available');
                return;
            }

            const jsonDisplay = document.getElementById('jsonDisplay');
            jsonDisplay.innerHTML = syntaxHighlight(JSON.stringify(currentAnalysisData, null, 2));
            document.getElementById('rawDataModal').classList.add('active');
        }

        function closeRawDataModal() {
            document.getElementById('rawDataModal').classList.remove('active');
        }

        function syntaxHighlight(json) {
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\\\u[a-zA-Z0-9]{4}|\\\\[^u]|[^\\\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?)/g, function (match) {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }

        const uploadArea = document.querySelector('.file-upload-area');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(0, 212, 255, 0.6)';
            uploadArea.style.background = 'rgba(0, 212, 255, 0.12)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(0, 212, 255, 0.25)';
            uploadArea.style.background = 'rgba(0, 212, 255, 0.03)';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(0, 212, 255, 0.25)';
            uploadArea.style.background = 'rgba(0, 212, 255, 0.03)';

            const file = e.dataTransfer.files[0];
            if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx'))) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                document.getElementById('fileInput').files = dataTransfer.files;
                handleFileSelect({ target: { files: [file] } });
            } else {
                alert('Please upload a CSV or XLSX file');
            }
        });

        document.getElementById('uploadModal').addEventListener('click', (e) => {
            if (e.target.id === 'uploadModal') {
                closeUploadModal();
            }
        });

        document.getElementById('rawDataModal').addEventListener('click', (e) => {
            if (e.target.id === 'rawDataModal') {
                closeRawDataModal();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeUploadModal();
                closeRawDataModal();
            }
        });

        window.addEventListener('DOMContentLoaded', createParticles);
    </script>
</body>
</html>
"""

    return html_content


def main():
    """Main execution function"""
    print("=" * 70)
    print("🚀 PlanetExplorer - FINAL VERSION")
    print("   NASA Space Apps Challenge 2025")
    print("=" * 70)
    print()

    html_content = generate_html()

    output_file = "planetexplorer_final.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Successfully generated {output_file}")
    print(f"📦 File size: {len(html_content):,} bytes")
    print()
    print("🎨 NEW FEATURES:")
    print("   ✓ Enhanced loading screen with:")
    print("     - Animated telescope icon")
    print("     - Multi-layer spinning rings")
    print("     - Pulsing orb at center")
    print("     - Progress bar animation")
    print("     - Stage-by-stage updates")
    print("     - Floating analysis icons")
    print("   ✓ Download Report (generates .txt file)")
    print("   ✓ View Raw Data (displays JSON with syntax highlighting)")
    print("   ✓ Renamed 'Begin Analysis' to 'Begin Exploring'")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
